"""Gene Ontology graph operations, propagation, and information content."""

from pathlib import Path

import networkx as nx
import obonet


# Namespace mapping: aspect code -> OBO namespace
# The dataset uses single-letter codes: P (Biological Process), F (Molecular Function), C (Cellular Component)
ASPECT_TO_NAMESPACE = {
    "P": "biological_process",
    "F": "molecular_function",
    "C": "cellular_component",
}
NAMESPACE_TO_ASPECT = {v: k for k, v in ASPECT_TO_NAMESPACE.items()}

# Full name mapping for display
ASPECT_FULL_NAME = {
    "P": "Biological Process (BPO)",
    "F": "Molecular Function (MFO)",
    "C": "Cellular Component (CCO)",
}


def load_go_graph(obo_path: str | Path) -> nx.DiGraph:
    """Load GO ontology from OBO file into a directed graph.

    Edges go from child -> parent (is_a relationships).
    Returns a DiGraph (not MultiDiGraph) for simpler traversal.
    """
    graph = obonet.read_obo(str(obo_path))
    # obonet returns a MultiDiGraph; convert to DiGraph keeping only is_a edges
    di = nx.DiGraph()
    for node, data in graph.nodes(data=True):
        if not node.startswith("GO:"):
            continue
        di.add_node(node, **data)
    for u, v, data in graph.edges(data=True):
        if not u.startswith("GO:") or not v.startswith("GO:"):
            continue
        di.add_edge(u, v)
    return di


def get_ontology_terms(graph: nx.DiGraph, aspect: str) -> set[str]:
    """Get all GO term IDs for a given ontology aspect (P, F, or C)."""
    namespace = ASPECT_TO_NAMESPACE[aspect]
    return {
        node
        for node, data in graph.nodes(data=True)
        if data.get("namespace") == namespace
    }


def get_ancestors(graph: nx.DiGraph, term_id: str) -> set[str]:
    """Get all ancestors of a GO term (including itself).

    Since edges go child -> parent, ancestors are descendants in graph direction.
    """
    ancestors = nx.descendants(graph, term_id)  # follows edges child->parent
    ancestors.add(term_id)
    return ancestors


def propagate_annotations(
    protein_terms: dict[str, set[str]], graph: nx.DiGraph
) -> dict[str, set[str]]:
    """Propagate annotations up the GO DAG (true path rule).

    For each protein, add all ancestor terms of its annotated terms.
    """
    propagated = {}
    for protein_id, terms in protein_terms.items():
        all_terms = set()
        for term in terms:
            if term in graph:
                all_terms.update(get_ancestors(graph, term))
            else:
                all_terms.add(term)
        propagated[protein_id] = all_terms
    return propagated


def propagate_scores(
    protein_scores: dict[str, dict[str, float]], graph: nx.DiGraph
) -> dict[str, dict[str, float]]:
    """Propagate prediction scores up the GO DAG.

    For each predicted term, ensure all ancestor terms have at least
    the same confidence score (take max of existing and propagated).
    """
    propagated = {}
    for protein_id, scores in protein_scores.items():
        new_scores = dict(scores)
        for term, conf in scores.items():
            if term in graph:
                for ancestor in get_ancestors(graph, term):
                    new_scores[ancestor] = max(new_scores.get(ancestor, 0.0), conf)
        propagated[protein_id] = new_scores
    return propagated


def compute_ic(term_counts: dict[str, int], total_proteins: int) -> dict[str, float]:
    """Compute Information Content for each GO term.

    IC(t) = -log2(frequency(t))
    """
    import numpy as np

    ic = {}
    for term, count in term_counts.items():
        freq = count / total_proteins
        if freq > 0:
            ic[term] = -np.log2(freq)
        else:
            ic[term] = 0.0
    return ic


def get_term_depth(graph: nx.DiGraph, term_id: str) -> int:
    """Get the depth of a term in the GO DAG (longest path to a root)."""
    roots = [n for n in graph.nodes() if graph.out_degree(n) == 0]
    max_depth = 0
    for root in roots:
        try:
            paths = nx.all_simple_paths(graph, term_id, root)
            for path in paths:
                max_depth = max(max_depth, len(path) - 1)
        except nx.NetworkXError:
            continue
    return max_depth
