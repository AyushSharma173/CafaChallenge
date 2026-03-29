#!/usr/bin/env python3
"""Generate a submission CSV from trained models.

Usage:
    python scripts/submit.py
    python scripts/submit.py --model-type mlp
    python scripts/submit.py --output submissions/my_submission.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from cafa6.config import Config
from cafa6.data_loader import load_fasta
from cafa6.embeddings import load_embeddings
from cafa6.go_utils import load_go_graph
from cafa6.models import BaseModel
from cafa6.submission import generate_submission, predictions_from_matrices, validate_submission


def main():
    parser = argparse.ArgumentParser(description="Generate submission")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    model_type = args.model_type or cfg.model_type

    # Load test embeddings
    emb_file = Path(cfg.embeddings_dir) / f"test_{cfg.embeddings.model_name}.h5"
    embeddings, protein_ids = load_embeddings(emb_file)
    print(f"Loaded {len(protein_ids)} test embeddings")

    # Load GO graph
    go_graph = load_go_graph(Path(cfg.data_dir) / "Train" / "go-basic.obo")

    # Generate predictions per ontology
    score_matrices = {}
    for aspect in cfg.ontologies:
        model_path = Path(cfg.models_dir) / f"{model_type}_{aspect}.pkl"
        term_ids_path = Path(cfg.models_dir) / f"term_ids_{aspect}.npy"

        if not model_path.exists():
            print(f"Model not found for {aspect}: {model_path}")
            continue

        model = BaseModel.load(model_path)
        term_ids = list(np.load(term_ids_path, allow_pickle=True))

        scores = model.predict(embeddings)
        score_matrices[aspect] = (scores, protein_ids, term_ids)
        print(f"{aspect}: predicted {scores.shape}")

    # Convert to submission format
    predictions = predictions_from_matrices(score_matrices)

    # Generate submission
    output_path = args.output or str(
        Path(cfg.submissions_dir) / f"{model_type}_submission.csv"
    )
    df = generate_submission(
        predictions,
        go_graph=go_graph,
        output_path=output_path,
        propagate=cfg.propagate,
        min_confidence=cfg.min_confidence,
    )

    # Validate
    test_fasta = Path(cfg.data_dir) / "Test" / "testsuperset.fasta"
    test_ids = None
    if test_fasta.exists():
        test_ids = set(load_fasta(test_fasta).keys())
    valid_terms = set(go_graph.nodes())
    validate_submission(output_path, test_ids, valid_terms)


if __name__ == "__main__":
    main()
