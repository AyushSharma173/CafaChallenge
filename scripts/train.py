#!/usr/bin/env python3
"""Train a model on precomputed embeddings.

Usage:
    python scripts/train.py                           # LightGBM (default)
    python scripts/train.py --model-type mlp          # MLP
    python scripts/train.py --aspect BPO              # Single ontology
    python scripts/train.py --config configs/custom.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from cafa6.config import Config
from cafa6.data_loader import (
    build_label_matrix,
    create_cv_split,
    load_taxonomy,
    load_train_terms,
)
from cafa6.embeddings import load_embeddings
from cafa6.go_utils import load_go_graph
from cafa6.metrics import compute_fmax
from cafa6.models import LightGBMMultilabel, MLPMultilabel, NaiveFrequency


def main():
    parser = argparse.ArgumentParser(description="Train CAFA6 model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-type", default=None, help="Override model type")
    parser.add_argument("--aspect", default=None, help="Train single ontology (P/F/C)")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    model_type = args.model_type or cfg.model_type
    aspects = [args.aspect] if args.aspect else cfg.ontologies

    # Load data
    print("Loading data...")
    go_graph = load_go_graph(Path(cfg.data_dir) / "Train" / "go-basic.obo")
    terms_df = load_train_terms(Path(cfg.data_dir) / "Train" / "train_terms.tsv")
    taxonomy_df = load_taxonomy(Path(cfg.data_dir) / "Train" / "train_taxonomy.tsv")

    emb_file = Path(cfg.embeddings_dir) / f"train_{cfg.embeddings.model_name}.h5"
    embeddings, emb_ids = load_embeddings(emb_file)
    emb_lookup = {pid: emb for pid, emb in zip(emb_ids, embeddings)}
    print(f"Loaded {len(emb_lookup)} embeddings (dim={embeddings.shape[1]})")

    models_dir = Path(cfg.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    for aspect in aspects:
        print(f"\n{'='*60}")
        print(f"Training {model_type} for {aspect}")
        print(f"{'='*60}")

        # Build label matrix
        label_matrix, protein_ids, term_ids = build_label_matrix(
            terms_df, go_graph, aspect, min_count=cfg.min_term_count
        )
        print(f"Label matrix: {label_matrix.shape[0]} proteins x {label_matrix.shape[1]} terms")

        # Filter to proteins with embeddings
        has_emb = [i for i, pid in enumerate(protein_ids) if pid in emb_lookup]
        protein_ids_f = [protein_ids[i] for i in has_emb]
        label_matrix_f = label_matrix[has_emb]
        X = np.array([emb_lookup[pid] for pid in protein_ids_f], dtype=np.float32)
        print(f"After embedding filter: {X.shape[0]} proteins")

        # Train/val split
        train_ids, val_ids = create_cv_split(
            protein_ids_f, taxonomy_df, cfg.val_fraction, cfg.seed
        )
        train_mask = np.array([pid in set(train_ids) for pid in protein_ids_f])
        val_mask = ~train_mask

        X_train, X_val = X[train_mask], X[val_mask]
        Y_train = label_matrix_f[train_mask]
        Y_val = label_matrix_f[val_mask]
        print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

        # Train model
        if model_type == "frequency":
            model = NaiveFrequency()
            model.fit(X_train, Y_train)
        elif model_type == "lightgbm":
            model = LightGBMMultilabel(
                n_estimators=cfg.lightgbm.n_estimators,
                learning_rate=cfg.lightgbm.learning_rate,
                num_leaves=cfg.lightgbm.num_leaves,
                min_child_samples=cfg.lightgbm.min_child_samples,
            )
            model.fit(X_train, Y_train)
        elif model_type == "mlp":
            model = MLPMultilabel(
                input_dim=X.shape[1],
                hidden_dims=cfg.mlp.hidden_dims,
                dropout=cfg.mlp.dropout,
                lr=cfg.mlp.lr,
                epochs=cfg.mlp.epochs,
                batch_size=cfg.mlp.batch_size,
            )
            model.fit(X_train, Y_train, X_val, Y_val)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Evaluate
        val_scores = model.predict(X_val)
        fmax, threshold = compute_fmax(Y_val, val_scores)
        print(f"\n{aspect} Validation Fmax: {fmax:.4f} (threshold={threshold:.2f})")

        # Save
        save_path = models_dir / f"{model_type}_{aspect}.pkl"
        model.save(save_path)
        print(f"Saved to {save_path}")

        # Also save term_ids for submission generation
        np.save(models_dir / f"term_ids_{aspect}.npy", term_ids)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
