"""Submission generation with GO propagation and format validation."""

from pathlib import Path

import pandas as pd

from .go_utils import propagate_scores


def generate_submission(
    predictions: dict[str, dict[str, float]],
    go_graph=None,
    output_path: str | Path = "submission.csv",
    propagate: bool = True,
    min_confidence: float = 0.01,
) -> pd.DataFrame:
    """Generate a submission CSV from predictions.

    Args:
        predictions: {protein_id: {GO_term: confidence_score}}
        go_graph: NetworkX DiGraph (required if propagate=True)
        output_path: Where to write the CSV
        propagate: Whether to propagate scores up the GO DAG
        min_confidence: Minimum confidence to include

    Returns:
        DataFrame of the submission
    """
    if propagate:
        if go_graph is None:
            raise ValueError("go_graph required when propagate=True")
        predictions = propagate_scores(predictions, go_graph)

    rows = []
    for protein_id, scores in predictions.items():
        for term, conf in scores.items():
            if conf >= min_confidence:
                rows.append((protein_id, term, conf))

    df = pd.DataFrame(rows, columns=["protein_id", "term", "confidence"])
    df = df.sort_values(["protein_id", "confidence"], ascending=[True, False])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, header=True)

    print(f"Submission saved: {output_path}")
    print(f"  Proteins: {df['protein_id'].nunique()}")
    print(f"  Total predictions: {len(df)}")
    print(f"  Avg predictions/protein: {len(df) / max(df['protein_id'].nunique(), 1):.1f}")

    return df


def validate_submission(
    submission_path: str | Path,
    test_protein_ids: set[str] | None = None,
    valid_go_terms: set[str] | None = None,
) -> list[str]:
    """Validate a submission file format. Returns list of warnings/errors."""
    issues = []
    df = pd.read_csv(submission_path)

    # Check columns
    expected_cols = {"protein_id", "term", "confidence"}
    if set(df.columns) != expected_cols:
        issues.append(f"Expected columns {expected_cols}, got {set(df.columns)}")
        return issues

    # Check confidence range
    if (df["confidence"] < 0).any() or (df["confidence"] > 1).any():
        issues.append("Confidence scores must be in [0, 1]")

    # Check for duplicates
    dupes = df.duplicated(subset=["protein_id", "term"])
    if dupes.any():
        issues.append(f"Found {dupes.sum()} duplicate (protein, term) pairs")

    # Check protein IDs
    if test_protein_ids is not None:
        unknown = set(df["protein_id"]) - test_protein_ids
        if unknown:
            issues.append(f"{len(unknown)} protein IDs not in test set")

    # Check GO terms
    if valid_go_terms is not None:
        unknown_terms = set(df["term"]) - valid_go_terms
        if unknown_terms:
            issues.append(f"{len(unknown_terms)} unknown GO terms")

    # Check GO term format
    bad_format = ~df["term"].str.match(r"^GO:\d{7}$")
    if bad_format.any():
        issues.append(f"{bad_format.sum()} terms don't match GO:NNNNNNN format")

    if not issues:
        print("Submission validation passed!")
    else:
        for issue in issues:
            print(f"WARNING: {issue}")

    return issues


def predictions_from_matrices(
    score_matrices: dict[str, tuple],
    threshold: float | None = None,
) -> dict[str, dict[str, float]]:
    """Convert per-ontology score matrices to prediction dict.

    Args:
        score_matrices: {aspect: (scores_array, protein_ids, term_ids)}
            scores_array: shape (n_proteins, n_terms) with confidence scores
        threshold: Optional minimum threshold (if None, include all > 0)

    Returns:
        {protein_id: {GO_term: confidence}}
    """
    predictions: dict[str, dict[str, float]] = {}

    for aspect, (scores, protein_ids, term_ids) in score_matrices.items():
        for i, pid in enumerate(protein_ids):
            if pid not in predictions:
                predictions[pid] = {}
            for j, tid in enumerate(term_ids):
                conf = float(scores[i, j])
                if threshold is not None and conf < threshold:
                    continue
                if conf > 0:
                    predictions[pid][tid] = max(predictions[pid].get(tid, 0), conf)

    return predictions
