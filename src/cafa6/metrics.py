"""Evaluation metrics matching CAFA's official scoring.

Key metric: Fmax (maximum protein-centric F1 across thresholds).
Secondary: Smin (minimum semantic distance using IA weights).
"""

import numpy as np
from scipy import sparse


def compute_fmax(
    y_true: np.ndarray | sparse.spmatrix,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Compute Fmax: maximum F1 across classification thresholds.

    This is PROTEIN-CENTRIC: precision and recall are computed per protein
    and averaged across proteins, then F1 is computed from averaged P/R.

    Args:
        y_true: Binary ground truth, shape (n_proteins, n_terms). Sparse OK.
        y_score: Predicted confidence scores, shape (n_proteins, n_terms).
        thresholds: Thresholds to evaluate. Defaults to 0.01 to 0.99.

    Returns:
        (fmax, best_threshold)
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    if sparse.issparse(y_true):
        y_true = y_true.toarray()

    n_proteins = y_true.shape[0]

    # Only evaluate proteins that have at least one annotation
    has_annotation = y_true.sum(axis=1) > 0
    y_true_f = y_true[has_annotation]
    y_score_f = y_score[has_annotation]
    n_eval = y_true_f.shape[0]

    if n_eval == 0:
        return 0.0, 0.0

    best_f1 = 0.0
    best_t = 0.0

    for t in thresholds:
        y_pred = (y_score_f >= t).astype(np.float32)

        # Per-protein precision and recall
        tp = (y_pred * y_true_f).sum(axis=1)
        pred_pos = y_pred.sum(axis=1)
        true_pos = y_true_f.sum(axis=1)

        # Precision: only for proteins with at least one prediction
        has_pred = pred_pos > 0
        if has_pred.sum() == 0:
            continue

        precision_per_protein = np.zeros(n_eval)
        precision_per_protein[has_pred] = tp[has_pred] / pred_pos[has_pred]

        recall_per_protein = tp / true_pos  # true_pos > 0 guaranteed by filter

        # Average across proteins (precision only over proteins with predictions)
        avg_precision = precision_per_protein[has_pred].mean()
        avg_recall = recall_per_protein.mean()

        if avg_precision + avg_recall == 0:
            continue

        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_f1, best_t


def compute_smin(
    y_true: np.ndarray | sparse.spmatrix,
    y_score: np.ndarray,
    ia_weights: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Compute Smin: minimum remaining uncertainty using IA weights.

    Args:
        y_true: Binary ground truth, shape (n_proteins, n_terms).
        y_score: Predicted confidence scores, shape (n_proteins, n_terms).
        ia_weights: Information accretion weight per term, shape (n_terms,).
        thresholds: Thresholds to evaluate.

    Returns:
        (smin, best_threshold)
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    if sparse.issparse(y_true):
        y_true = y_true.toarray()

    has_annotation = y_true.sum(axis=1) > 0
    y_true_f = y_true[has_annotation]
    y_score_f = y_score[has_annotation]

    if y_true_f.shape[0] == 0:
        return 0.0, 0.0

    best_s = float("inf")
    best_t = 0.0

    for t in thresholds:
        y_pred = (y_score_f >= t).astype(np.float32)

        # Remaining uncertainty: weighted false negatives
        fn = y_true_f * (1 - y_pred)  # missed true annotations
        ru = (fn * ia_weights).sum(axis=1)

        # Misinformation: weighted false positives
        fp = (1 - y_true_f) * y_pred  # incorrect predictions
        mi = (fp * ia_weights).sum(axis=1)

        # Semantic distance per protein
        s_per_protein = np.sqrt(ru**2 + mi**2)
        avg_s = s_per_protein.mean()

        if avg_s < best_s:
            best_s = avg_s
            best_t = t

    return best_s, best_t


def evaluate_per_ontology(
    y_true_dict: dict[str, np.ndarray | sparse.spmatrix],
    y_score_dict: dict[str, np.ndarray],
    ia_weights_dict: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate Fmax (and optionally Smin) per ontology.

    Args:
        y_true_dict: {aspect: label_matrix} for each ontology
        y_score_dict: {aspect: score_matrix} for each ontology
        ia_weights_dict: {aspect: weights_array} (optional, for Smin)

    Returns:
        {aspect: {"fmax": float, "threshold": float, "smin": float (if available)}}
    """
    results = {}
    for aspect in y_true_dict:
        fmax, fmax_t = compute_fmax(y_true_dict[aspect], y_score_dict[aspect])
        entry = {"fmax": fmax, "fmax_threshold": fmax_t}

        if ia_weights_dict and aspect in ia_weights_dict:
            smin, smin_t = compute_smin(
                y_true_dict[aspect], y_score_dict[aspect], ia_weights_dict[aspect]
            )
            entry["smin"] = smin
            entry["smin_threshold"] = smin_t

        results[aspect] = entry

    # Overall average
    fmax_values = [r["fmax"] for r in results.values()]
    results["overall"] = {"fmax": np.mean(fmax_values)}

    return results
