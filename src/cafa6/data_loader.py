"""Data loading utilities for FASTA, TSV, OBO files and label matrix construction."""

from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy import sparse
from sklearn.model_selection import GroupShuffleSplit

from .go_utils import get_ancestors, get_ontology_terms


def load_fasta(path: str | Path) -> dict[str, str]:
    """Load protein sequences from a FASTA file.

    Handles two header formats:
      - Train: >sp|ACCESSION|NAME description
      - Test:  >ACCESSION taxon_id

    Returns: {protein_id: amino_acid_sequence}
    """
    sequences = {}
    for record in SeqIO.parse(str(path), "fasta"):
        rid = record.id
        if "|" in rid:
            # Train format: sp|A0A0C5B5G6|MOTSC_HUMAN -> extract accession
            protein_id = rid.split("|")[1]
        else:
            # Test format: A0A0C5B5G6 -> already the ID
            protein_id = rid
        sequences[protein_id] = str(record.seq)
    return sequences


def load_train_terms(path: str | Path) -> pd.DataFrame:
    """Load training GO annotations.

    Returns DataFrame with columns: [EntryID, term, aspect]
    Aspect values are single letters: P (BPO), F (MFO), C (CCO)
    """
    df = pd.read_csv(str(path), sep="\t")
    return df


def load_taxonomy(path: str | Path) -> pd.DataFrame:
    """Load protein-to-species taxonomy mapping.

    Returns DataFrame with columns: [EntryID, taxonomyID]
    """
    df = pd.read_csv(str(path), sep="\t", header=None, names=["EntryID", "taxonomyID"])
    return df


def load_ia_weights(path: str | Path) -> dict[str, float]:
    """Load information accretion weights for GO terms.

    Returns: {GO_term: IA_weight}
    """
    weights = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                weights[parts[0]] = float(parts[1])
    return weights


def build_label_matrix(
    terms_df: pd.DataFrame,
    go_graph,
    aspect: str,
    min_count: int = 50,
    propagate: bool = True,
) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    """Build a sparse binary label matrix for one ontology.

    Args:
        terms_df: DataFrame with [EntryID, term, aspect]
        go_graph: NetworkX DiGraph of the GO ontology
        aspect: One of "P", "F", "C"
        min_count: Minimum number of proteins a term must appear in
        propagate: Whether to propagate annotations up the GO DAG

    Returns:
        (label_matrix, protein_ids, term_ids)
        label_matrix: sparse binary matrix of shape (n_proteins, n_terms)
    """
    # Filter to the target ontology
    aspect_df = terms_df[terms_df["aspect"] == aspect].copy()

    # Group annotations by protein
    protein_terms: dict[str, set[str]] = {}
    for _, row in aspect_df.iterrows():
        protein_terms.setdefault(row["EntryID"], set()).add(row["term"])

    # Propagate annotations up DAG
    if propagate:
        ontology_terms = get_ontology_terms(go_graph, aspect)
        propagated = {}
        for pid, terms in protein_terms.items():
            all_terms = set()
            for t in terms:
                if t in go_graph:
                    ancestors = get_ancestors(go_graph, t)
                    all_terms.update(ancestors & ontology_terms)
                else:
                    all_terms.add(t)
            propagated[pid] = all_terms
        protein_terms = propagated

    # Count term frequencies
    term_counts: dict[str, int] = {}
    for terms in protein_terms.values():
        for t in terms:
            term_counts[t] = term_counts.get(t, 0) + 1

    # Filter by min_count
    valid_terms = sorted([t for t, c in term_counts.items() if c >= min_count])
    term_to_idx = {t: i for i, t in enumerate(valid_terms)}

    # Build sparse matrix
    protein_ids = sorted(protein_terms.keys())
    pid_to_idx = {p: i for i, p in enumerate(protein_ids)}

    rows, cols = [], []
    for pid, terms in protein_terms.items():
        pi = pid_to_idx[pid]
        for t in terms:
            if t in term_to_idx:
                rows.append(pi)
                cols.append(term_to_idx[t])

    data = np.ones(len(rows), dtype=np.float32)
    label_matrix = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(protein_ids), len(valid_terms))
    )

    return label_matrix, protein_ids, valid_terms


def create_cv_split(
    protein_ids: list[str],
    taxonomy_df: pd.DataFrame | None = None,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Create train/validation split, optionally grouping by species.

    If taxonomy_df is provided, splits by species (GroupShuffleSplit)
    to avoid data leakage from related proteins.

    Returns: (train_ids, val_ids)
    """
    if taxonomy_df is not None:
        # Map protein_ids to taxonomy
        tax_map = dict(zip(taxonomy_df["EntryID"], taxonomy_df["taxonomyID"]))
        groups = [tax_map.get(pid, -1) for pid in protein_ids]

        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(splitter.split(protein_ids, groups=groups))
        train_ids = [protein_ids[i] for i in train_idx]
        val_ids = [protein_ids[i] for i in val_idx]
    else:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(protein_ids))
        n_val = int(len(protein_ids) * val_fraction)
        val_ids = [protein_ids[i] for i in indices[:n_val]]
        train_ids = [protein_ids[i] for i in indices[n_val:]]

    return train_ids, val_ids
