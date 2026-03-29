"""ESM-2 protein embedding extraction with GPU support and HDF5 storage."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm


def load_esm_model(model_name: str = "esm2_t33_650M_UR50D", device: str = "cuda"):
    """Load an ESM-2 model and return (model, alphabet, batch_converter)."""
    import esm

    model, alphabet = getattr(esm.pretrained, model_name)()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def extract_embeddings(
    sequences: dict[str, str],
    model,
    alphabet,
    batch_converter,
    repr_layer: int = 33,
    batch_size: int = 4,
    max_seq_len: int = 1022,
    device: str = "cuda",
    output_path: str = "embeddings.h5",
    checkpoint_every: int = 1000,
):
    """Extract mean-pooled ESM-2 embeddings for all sequences.

    Saves to HDF5 incrementally. Sequences are sorted by length for
    efficient batching (minimizes padding waste).

    Args:
        sequences: {protein_id: amino_acid_sequence}
        model: ESM-2 model
        alphabet: ESM alphabet
        batch_converter: Batch converter function
        repr_layer: Which transformer layer to extract from
        batch_size: Sequences per batch (adjust by GPU VRAM)
        max_seq_len: Max tokens (ESM-2 limit is 1022)
        device: "cuda", "cpu", or "mps"
        output_path: Path for HDF5 output file
        checkpoint_every: Save progress every N proteins
    """
    # Sort by length for efficient batching
    sorted_items = sorted(sequences.items(), key=lambda x: len(x[1]))

    # Check for existing progress
    completed_ids = set()
    output_path = Path(output_path)
    if output_path.exists():
        with h5py.File(output_path, "r") as f:
            if "protein_ids" in f:
                completed_ids = set(f["protein_ids"].asstr()[:])
        print(f"Resuming: {len(completed_ids)} proteins already processed")

    # Filter out completed
    remaining = [(pid, seq) for pid, seq in sorted_items if pid not in completed_ids]
    if not remaining:
        print("All proteins already processed!")
        return

    print(f"Processing {len(remaining)} proteins...")

    # Get embedding dimension from a test forward pass
    test_data = [("test", "MKTL")]
    _, _, test_tokens = batch_converter(test_data)
    with torch.no_grad():
        test_out = model(test_tokens.to(device), repr_layers=[repr_layer])
    embed_dim = test_out["representations"][repr_layer].shape[-1]
    del test_out, test_tokens

    # Collect results in memory, flush periodically
    batch_ids = []
    batch_embeds = []

    def flush_to_hdf5():
        if not batch_ids:
            return
        mode = "a" if output_path.exists() else "w"
        with h5py.File(output_path, mode) as f:
            if "embeddings" not in f:
                f.create_dataset(
                    "embeddings",
                    shape=(0, embed_dim),
                    maxshape=(None, embed_dim),
                    dtype="float16",
                    chunks=(min(1000, len(sequences)), embed_dim),
                )
                dt = h5py.special_dtype(vlen=str)
                f.create_dataset("protein_ids", shape=(0,), maxshape=(None,), dtype=dt)

            emb_ds = f["embeddings"]
            id_ds = f["protein_ids"]
            n = emb_ds.shape[0]
            n_new = len(batch_ids)

            emb_ds.resize(n + n_new, axis=0)
            id_ds.resize(n + n_new, axis=0)

            emb_ds[n : n + n_new] = np.array(batch_embeds, dtype=np.float16)
            id_ds[n : n + n_new] = batch_ids

        batch_ids.clear()
        batch_embeds.clear()

    # Process in batches
    pbar = tqdm(total=len(remaining), desc="Extracting embeddings")

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i : i + batch_size]

        # Truncate long sequences
        data = [(pid, seq[:max_seq_len]) for pid, seq in batch]

        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[repr_layer])

        representations = results["representations"][repr_layer]

        for j, (pid, seq) in enumerate(data):
            # Mean pool over sequence (exclude BOS at 0 and EOS/padding)
            seq_len = min(len(seq), max_seq_len)
            embedding = representations[j, 1 : seq_len + 1].mean(dim=0).cpu().numpy()
            batch_ids.append(pid)
            batch_embeds.append(embedding)

        pbar.update(len(batch))

        # Checkpoint
        if len(batch_ids) >= checkpoint_every:
            flush_to_hdf5()
            pbar.set_postfix(saved=True)

        del tokens, results, representations
        if device == "cuda":
            torch.cuda.empty_cache()

    # Final flush
    flush_to_hdf5()
    pbar.close()
    print(f"Saved embeddings to {output_path}")


def load_embeddings(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Load embeddings from HDF5 file.

    Returns: (embeddings_array [n_proteins, embed_dim], protein_ids)
    """
    with h5py.File(str(path), "r") as f:
        embeddings = f["embeddings"][:]
        protein_ids = list(f["protein_ids"].asstr()[:])
    return embeddings, protein_ids


def main():
    """CLI entry point for embedding extraction."""
    parser = argparse.ArgumentParser(description="Extract ESM-2 embeddings")
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--output", required=True, help="Output HDF5 file")
    parser.add_argument("--model", default="esm2_t33_650M_UR50D", help="ESM-2 model name")
    parser.add_argument("--repr-layer", type=int, default=33, help="Representation layer")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-seq-len", type=int, default=1022, help="Max sequence length")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu/mps)")
    args = parser.parse_args()

    from .data_loader import load_fasta

    sequences = load_fasta(args.fasta)
    print(f"Loaded {len(sequences)} sequences")

    model, alphabet, batch_converter = load_esm_model(args.model, args.device)

    extract_embeddings(
        sequences=sequences,
        model=model,
        alphabet=alphabet,
        batch_converter=batch_converter,
        repr_layer=args.repr_layer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
