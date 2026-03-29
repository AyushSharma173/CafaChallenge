#!/usr/bin/env python3
"""Generate ESM-2 embeddings for train and test proteins.

Usage:
    python scripts/generate_embeddings.py                    # defaults
    python scripts/generate_embeddings.py --device mps       # Apple Silicon
    python scripts/generate_embeddings.py --batch-size 8     # larger GPU
    python scripts/generate_embeddings.py --model esm2_t36_3B_UR50D  # bigger model
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cafa6.config import Config
from cafa6.data_loader import load_fasta
from cafa6.embeddings import extract_embeddings, load_esm_model


def main():
    parser = argparse.ArgumentParser(description="Generate ESM-2 embeddings")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--device", default=None, help="Override device (cuda/mps/cpu)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--model", default=None, help="Override ESM model name")
    parser.add_argument("--train-only", action="store_true", help="Only process train sequences")
    parser.add_argument("--test-only", action="store_true", help="Only process test sequences")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    model_name = args.model or cfg.embeddings.model_name
    batch_size = args.batch_size or cfg.embeddings.batch_size
    device = args.device
    if device is None:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Model: {model_name}, Device: {device}, Batch size: {batch_size}")
    model, alphabet, batch_converter = load_esm_model(model_name, device)

    embeddings_dir = Path(cfg.embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Process train sequences
    if not args.test_only:
        train_fasta = Path(cfg.data_dir) / "Train" / "train_sequences.fasta"
        if train_fasta.exists():
            print(f"\n=== Processing training sequences ===")
            sequences = load_fasta(train_fasta)
            print(f"Loaded {len(sequences)} training sequences")
            extract_embeddings(
                sequences=sequences,
                model=model,
                alphabet=alphabet,
                batch_converter=batch_converter,
                repr_layer=cfg.embeddings.repr_layer,
                batch_size=batch_size,
                max_seq_len=cfg.embeddings.max_seq_len,
                device=device,
                output_path=str(embeddings_dir / f"train_{model_name}.h5"),
            )
        else:
            print(f"Train FASTA not found: {train_fasta}")

    # Process test sequences
    if not args.train_only:
        test_fasta = Path(cfg.data_dir) / "Test" / "testsuperset.fasta"
        if test_fasta.exists():
            print(f"\n=== Processing test sequences ===")
            sequences = load_fasta(test_fasta)
            print(f"Loaded {len(sequences)} test sequences")
            extract_embeddings(
                sequences=sequences,
                model=model,
                alphabet=alphabet,
                batch_converter=batch_converter,
                repr_layer=cfg.embeddings.repr_layer,
                batch_size=batch_size,
                max_seq_len=cfg.embeddings.max_seq_len,
                device=device,
                output_path=str(embeddings_dir / f"test_{model_name}.h5"),
            )
        else:
            print(f"Test FASTA not found: {test_fasta}")

    print("\nDone!")


if __name__ == "__main__":
    main()
