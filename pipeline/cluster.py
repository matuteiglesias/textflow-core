# cluster.py
import numpy as np
import textwrap
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Optional

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import (
    linkage, leaves_list, fcluster, cophenet, optimal_leaf_ordering
)

# ---------- Core: linkage + optimal leaf order ----------
def _build_linkage(vecs: np.ndarray, *, metric="cosine", method="complete", optimal=True):
    """
    Build a hierarchical linkage. If optimal=True, apply optimal_leaf_ordering
    to minimize the sum of distances between adjacent leaves.
    """
    if len(vecs) == 0:
        return None, np.array([], dtype=int), None

    # distance vector (so we don’t recompute inside linkage)
    Y = pdist(vecs, metric=metric)
    Z = linkage(Y, method=method)
    if optimal:
        Z = optimal_leaf_ordering(Z, Y)
    leaf_order = leaves_list(Z)
    return Z, leaf_order, Y


# ---------- Public: cluster with dendrogram order preserved ----------
def cluster_embeddings(
    vecs: np.ndarray,
    nodes: List,
    *,
    method: str = "complete",
    metric: str = "cosine",
    t: Optional[float] = 0.35,
    maxclust: Optional[int] = None,
    optimal_order: bool = True,
):
    """
    Hierarchical clustering (cosine by default) with optimal leaf ordering.
    Returns nodes sorted by dendrogram leaves and labels aligned to that order.

    Parameters
    ----------
    t : float
        Distance cut threshold (cosine distance). Ignored if maxclust is set.
    maxclust : int
        If provided, use fcluster(..., criterion="maxclust") for predictable K.

    Returns
    -------
    ordered_nodes : List
    labels_ordered : np.ndarray
    Z : ndarray (linkage matrix)
    leaf_order : np.ndarray
    """
    assert len(vecs) == len(nodes), "vecs/nodes length mismatch"

    if len(nodes) == 0:
        return [], np.array([], dtype=int), None, np.array([], dtype=int)

    Z, leaf_order, Y = _build_linkage(
        vecs, metric=metric, method=method, optimal=optimal_order
    )

    # Cluster on the *optimized* linkage matrix
    if maxclust is not None:
        labels = fcluster(Z, t=maxclust, criterion="maxclust")
    else:
        labels = fcluster(Z, t=t, criterion="distance")

    # reorder by dendrogram leaves
    ordered_nodes  = [nodes[i] for i in leaf_order]
    labels_ordered = np.asarray(labels, dtype=int)[leaf_order]

    return ordered_nodes, labels_ordered, Z, leaf_order


# ---------- Optional: post-merge tiny clusters ----------
def merge_small_clusters(
    vecs: np.ndarray,
    ordered_labels: np.ndarray,
    min_size: int = 3
) -> np.ndarray:
    """
    Merge clusters with size < min_size into the nearest (cosine-sim) larger cluster.
    Works on labels *in leaf order*. Returns new labels in the same order.
    """
    labs = ordered_labels.copy()
    if len(labs) == 0:
        return labs

    # compute cluster centroids
    id2idxs: Dict[int, List[int]] = defaultdict(list)
    for i, c in enumerate(labs):
        id2idxs[c].append(i)

    big = [c for c, idxs in id2idxs.items() if len(idxs) >= min_size]
    small = [c for c, idxs in id2idxs.items() if len(idxs) < min_size]
    if not small or not big:
        return labs

    # pre-normalize vectors for cosine sim
    v = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    cent = {c: v[id2idxs[c]].mean(axis=0) for c in id2idxs}
    for c in cent:
        cent[c] = cent[c] / (np.linalg.norm(cent[c]) + 1e-9)

    for c in small:
        # nearest big centroid
        best = max(big, key=lambda B: float(cent[c] @ cent[B]))
        for i in id2idxs[c]:
            labs[i] = best

    return labs


# ---------- Grouping in dendrogram order ----------
def renumber_labels_in_order(labels_ordered: np.ndarray) -> np.ndarray:
    """
    Make cluster ids sequential by first-appearance along leaf order: 1,2,3,...
    """
    mapping = {}
    next_id = 1
    out = labels_ordered.copy()
    for i, c in enumerate(labels_ordered):
        if c not in mapping:
            mapping[c] = next_id
            next_id += 1
        out[i] = mapping[c]
    return out


def group_nodes_contiguous(ordered_nodes: List, labels_ordered: np.ndarray):
    """
    Group nodes as contiguous blocks along the leaf order.
    Normally each label forms one block after a dendrogram cut; this also
    handles rare split blocks gracefully.
    Returns a list of tuples: (label, [nodes...]) preserving leaf order.
    """
    groups: List[Tuple[int, List]] = []
    if not ordered_nodes:
        return groups

    cur_lab = int(labels_ordered[0])
    cur_bucket = [ordered_nodes[0]]

    for n, lab in zip(ordered_nodes[1:], labels_ordered[1:]):
        lab = int(lab)
        if lab == cur_lab:
            cur_bucket.append(n)
        else:
            groups.append((cur_lab, cur_bucket))
            cur_lab = lab
            cur_bucket = [n]
    groups.append((cur_lab, cur_bucket))
    return groups


# ---------- Printing / preview ----------
def preview_clusters_in_order(
    groups: Iterable[Tuple[int, List]],
    max_len: int = 100,
    metadata_field: str = "header_path"
):
    """
    Print clusters in dendrogram order. `groups` is the output of group_nodes_contiguous.
    """
    for lab, nodes in groups:
        print(f"\n## Cluster {lab} (n={len(nodes)})")
        for n in nodes:
            meta = n.metadata.get(metadata_field, "—") if hasattr(n, "metadata") else "—"
            txt  = getattr(n, "text", str(n))
            snippet = textwrap.shorten(txt.replace("\n", " "), max_len)
            print(f"{str(meta):40s} | {snippet}")


# ---------- (Optional) plain label->list grouping, but still leaf-ordered ----------
def group_nodes_by_label_in_leaf_order(
    ordered_nodes: List, labels_ordered: np.ndarray
) -> Dict[int, List]:
    """
    Groups all nodes by label but *preserves the leaf-order within each label*.
    Useful if you don’t want contiguous blocks, but still want local coherence.
    """
    d: Dict[int, List] = defaultdict(list)
    for n, lab in zip(ordered_nodes, labels_ordered):
        d[int(lab)].append(n)
    # keep label keys in first-appearance order
    order = []
    seen = set()
    for lab in labels_ordered:
        lab = int(lab)
        if lab not in seen:
            seen.add(lab)
            order.append(lab)
    return {lab: d[lab] for lab in order}
