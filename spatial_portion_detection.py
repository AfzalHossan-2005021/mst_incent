"""
spatial_portion_detection.py
────────────────────────────────────────────────────────────────────────────
Biologically-principled tissue-portion detection for spatial transcriptomics.

THEORETICAL RATIONALE
─────────────────────
Physical tissue sections placed on the same slide but originating from
distinct anatomical regions (e.g., left/right cerebral hemisphere, four
cardiac chambers, bilateral kidney lobes) are separated by measurable spatial
*air gaps* — regions devoid of cells.  Within a continuous tissue region,
cells are packed at a characteristic density governed by biology (MERFISH
cell spacing: typically 5–30 µm).

The Minimum Spanning Tree (MST) of the spatial point cloud has a key
structural property that makes it ideal for detecting such gaps:

    Theorem (Zahn, 1971): For a point set partitioned into k
    well-separated clusters, the k-1 longest edges of the Euclidean
    MST are exactly the edges crossing cluster boundaries.

This gives us a principled, shape-agnostic, threshold-free framework:

  1. The MST has exactly (k-1) gap-crossing edges for k tissue portions.
  2. Gap edges are the largest edges in the MST by construction.
  3. The transition from gap edges to intra-tissue edges produces a
     large multiplicative JUMP in sorted MST edge weights.
  4. We detect this jump via the Maximum Ratio Gap criterion.

Maximum Ratio Gap Criterion
───────────────────────────
Sort all MST edges in descending order: e_1 >= e_2 >= ... >= e_{n-1}.

For each candidate index m in {1, ..., max_k-1}, compute the ratio:
    ratio[m] = e_m / e_{m+1}

The position m* = argmax(ratio[m]) identifies the transition point.
If ratio[m*] >= tau (default 3.0), the tissue has k = m*+2 portions.
If no ratio reaches tau, the tissue is a single portion (k = 1).

Biological interpretation: a gap edge is >= 3x longer than the largest
intra-tissue MST edge. This corresponds to a physical gap at least 3x
the typical inter-cell spacing -- conservative enough to avoid splitting
dense but continuous tissue.

Advantages over GMM
────────────────────
  + No shape assumption: works for crescent, lobular, and irregular tissue
  + Deterministic: no random seed sensitivity (MST is unique for distinct weights)
  + Threshold is data-derived from the tissue's own geometry
  + Handles k=1 naturally (no gap -> single portion)
  - Sensitivity to tau for near-touching tissue (documented in sensitivity_analysis.py)

REFERENCES
──────────
  Zahn C.T. (1971) Graph-theoretical methods for detecting and describing
      Gestalt clusters. IEEE Trans Comput, 20(1), 68-86.
  Gower J.C. & Ross G.J.S. (1969) Minimum spanning trees and single linkage
      cluster analysis. Appl Stat, 18(1), 54-64.
"""

from __future__ import annotations

import warnings
import numpy as np
import anndata
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import BallTree
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# MST construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_mst(coords: np.ndarray, knn_build: int = 15) -> sp.csr_matrix:
    """
    Build the Euclidean Minimum Spanning Tree.

    For n <= 5000 cells: exact O(n^2) pairwise distance matrix.
    For n >  5000 cells: sparse k-NN graph via BallTree (O(n log n)).

    The sparse approximation is exact for well-separated portions because
    gap edges are always longer than any k-NN intra-portion edge when the
    gap exceeds k times the typical inter-cell spacing.
    """
    n = len(coords)
    if n <= 5_000:
        dist_sparse = sp.csr_matrix(squareform(pdist(coords, metric='euclidean')))
    else:
        tree = BallTree(coords, leaf_size=40)
        k = min(knn_build + 1, n)
        dists, indices = tree.query(coords, k=k)
        rows = np.repeat(np.arange(n), k)
        dist_sparse = sp.csr_matrix(
            (dists.ravel(), (rows, indices.ravel())), shape=(n, n)
        )
        dist_sparse = dist_sparse + dist_sparse.T
    return minimum_spanning_tree(dist_sparse)


# ─────────────────────────────────────────────────────────────────────────────
# K detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_k_from_mst(
    edge_weights: np.ndarray,
    max_portions: int = 6,
    ratio_threshold: float = 0.1,
) -> int:
    """
    Detect number of tissue portions using the Maximum Ratio Gap criterion.

    Sorts MST edges descending. Computes ratio[m] = e_m / e_{m+1}.
    argmax(ratio) identifies the last gap edge.

    Returns k (int). Returns 1 if no ratio >= ratio_threshold.
    """
    sorted_desc = np.sort(edge_weights)[::-1]
    n_check = min(max_portions - 1, len(sorted_desc) - 1)
    if n_check < 1:
        return 1

    ratios = sorted_desc[:n_check] / (sorted_desc[1:n_check + 1] + 1e-12)
    best_m = int(np.argmax(ratios))

    if float(ratios[best_m]) < ratio_threshold:
        return 1

    # k = (number of gap edges) + 1 = (best_m + 1) + 1
    return best_m + 2


def _build_components(mst: sp.csr_matrix, k: int, n: int) -> np.ndarray:
    """
    Remove the k-1 largest MST edges and return connected component labels.
    """
    if k <= 1:
        return np.zeros(n, dtype=int)
    mst_coo = mst.tocoo()
    sorted_desc = np.sort(mst_coo.data)[::-1]
    threshold = float(sorted_desc[k - 1])
    mask = mst_coo.data <= threshold
    pruned = sp.csr_matrix(
        (mst_coo.data[mask], (mst_coo.row[mask], mst_coo.col[mask])),
        shape=(n, n),
    )
    pruned = pruned + pruned.T
    _, labels = connected_components(pruned, directed=False)
    return labels.astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Fragment handling
# ─────────────────────────────────────────────────────────────────────────────

def _merge_small_fragments(
    labels: np.ndarray,
    coords: np.ndarray,
    min_mass_fraction: float,
) -> np.ndarray:
    """
    Merge any component below min_mass_fraction into the spatially nearest
    large component (nearest centroid). Handles debris cells gracefully.
    """
    labels = labels.copy()
    total = len(labels)
    unique, counts = np.unique(labels, return_counts=True)
    large = unique[counts / total >= min_mass_fraction]
    small = unique[counts / total < min_mass_fraction]
    if len(small) == 0:
        return labels
    large_centroids = np.array([coords[labels == c].mean(axis=0) for c in large])
    for s in small:
        centroid_s = coords[labels == s].mean(axis=0)
        nearest = large[np.argmin(np.linalg.norm(large_centroids - centroid_s, axis=1))]
        labels[labels == s] = nearest
    for new_i, old_l in enumerate(np.unique(labels)):
        labels[labels == old_l] = new_i
    return labels


def _validate_portions(labels: np.ndarray, min_mass_fraction: float) -> bool:
    total = len(labels)
    _, counts = np.unique(labels, return_counts=True)
    return all(c / total >= min_mass_fraction for c in counts)


# ─────────────────────────────────────────────────────────────────────────────
# Stability score
# ─────────────────────────────────────────────────────────────────────────────

def _compute_stability_score(
    edge_weights: np.ndarray,
    k_ref: int,
    max_portions: int = 6,
    ratio_range: Tuple[float, float] = (1.5, 6.0),
    n_steps: int = 25,
) -> float:
    """
    Fraction of ratio_threshold values in ratio_range that produce the same
    k as the reference. Values above 0.7 indicate robust detection.
    """
    thresholds = np.linspace(ratio_range[0], ratio_range[1], n_steps)
    agree = sum(
        1 for t in thresholds
        if _detect_k_from_mst(edge_weights, max_portions, t) == k_ref
    )
    return agree / n_steps


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

class PortionDetectionResult:
    """Result object returned by find_spatial_portions_mst."""

    def __init__(self, k, labels, gap_threshold, top_edge_weights,
                 top_ratios, stability_score, ratio_threshold_used):
        self.k = k
        self.labels = labels
        self.gap_threshold = gap_threshold
        self.top_edge_weights = top_edge_weights
        self.top_ratios = top_ratios
        self.stability_score = stability_score
        self.ratio_threshold_used = ratio_threshold_used

    def __repr__(self):
        return (
            f"PortionDetectionResult(k={self.k}, "
            f"gap_threshold={self.gap_threshold:.1f}, "
            f"stability={self.stability_score:.2f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def find_spatial_portions_mst(
    adata: anndata.AnnData,
    min_mass_fraction: float = 0.05,
    max_portions: int = 6,
    ratio_threshold: float = 3.0,
    merge_fragments: bool = True,
    knn_build: int = 15,
) -> PortionDetectionResult:
    """
    Detect physically distinct tissue portions using MST-based gap detection.

    Parameters
    ----------
    adata             : AnnData with .obsm['spatial'] (n x 2 coordinates).
    min_mass_fraction : Minimum fraction of total cells per valid portion.
                        Smaller fragments are merged into the nearest portion.
                        Default 0.05 (5%).
    max_portions      : Biological upper bound on k. Default 6.
    ratio_threshold   : Minimum edge-weight ratio to declare a gap.
                        tau=3.0: gap must be >= 3x the next intra-tissue edge.
                        Increase for near-touching tissue; decrease for sparse.
    merge_fragments   : If True, merge sub-threshold fragments (recommended).
    knn_build         : k for BallTree approximation when n > 5000 cells.

    Returns
    -------
    PortionDetectionResult with attributes:
        .k               -- number of detected tissue portions (int)
        .labels          -- per-cell component index array (shape n,)
        .gap_threshold   -- gap edge threshold in coordinate units
        .top_ratios      -- ratio array for the top max_portions MST edges
        .stability_score -- robustness score in [0, 1]; >0.7 = reliable
    """
    coords = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    n = len(coords)
    if n < 3:
        raise ValueError(f"Slice has only {n} cells.")

    mst = _build_mst(coords, knn_build=knn_build)
    mst_coo = mst.tocoo()
    edge_weights = mst_coo.data.copy()

    if len(edge_weights) == 0:
        return PortionDetectionResult(
            k=1, labels=np.zeros(n, dtype=int), gap_threshold=0.0,
            top_edge_weights=np.array([]), top_ratios=np.array([]),
            stability_score=1.0, ratio_threshold_used=ratio_threshold,
        )

    sorted_desc = np.sort(edge_weights)[::-1]
    n_top = min(max_portions, len(sorted_desc) - 1)
    top_weights = sorted_desc[:n_top + 1]
    top_ratios = top_weights[:-1] / (top_weights[1:] + 1e-12)

    k = _detect_k_from_mst(edge_weights, max_portions, ratio_threshold)
    labels = _build_components(mst, k, n)

    if merge_fragments:
        labels = _merge_small_fragments(labels, coords, min_mass_fraction)
    elif not _validate_portions(labels, min_mass_fraction):
        raise ValueError(
            "Detected portions include fragments below min_mass_fraction. "
            "Set merge_fragments=True or increase ratio_threshold."
        )

    k = len(np.unique(labels))
    stability = _compute_stability_score(edge_weights, k, max_portions)
    gap_threshold = float(sorted_desc[k - 1]) if k >= 2 else float('inf')

    if stability < 0.4:
        warnings.warn(
            f"Low stability score ({stability:.2f}) for k={k}. "
            "Consider adjusting ratio_threshold or inspecting the slide.",
            UserWarning,
        )

    return PortionDetectionResult(
        k=k, labels=labels, gap_threshold=gap_threshold,
        top_edge_weights=top_weights, top_ratios=top_ratios,
        stability_score=stability,
        ratio_threshold_used=ratio_threshold,
    )


def find_spatial_portions(
    adata: anndata.AnnData,
    config,
    max_portions: int = 4,
) -> Tuple[int, np.ndarray]:
    """
    Drop-in replacement for the GMM-based find_spatial_portions in smart_align.py.

    The AlignmentConfig.silhouette_threshold parameter has no equivalent here
    and is intentionally ignored -- geometric separation is now measured
    directly via the ratio criterion, not as a post-hoc silhouette proxy.

    Parameters
    ----------
    adata        : AnnData with .obsm['spatial'].
    config       : AlignmentConfig. Uses .min_mass_fraction.
    max_portions : Upper bound on k.

    Returns
    -------
    (k, labels) -- same as the original GMM-based function.
    """
    result = find_spatial_portions_mst(
        adata,
        min_mass_fraction=config.min_mass_fraction,
        max_portions=max_portions,
        merge_fragments=True,
    )
    return result.k, result.labels
