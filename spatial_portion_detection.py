"""
spatial_portion_detection.py
────────────────────────────────────────────────────────────────────────────
Biologically-principled tissue-portion detection for spatial transcriptomics.

THREE-DETECTOR CASCADED ENSEMBLE
─────────────────────────────────
Physical tissue sections on the same slide are separated by air gaps. No
single geometric detector is universally robust because real MERFISH/Xenium
data has varying gap widths, tissue densities, and edge bleed. Three
complementary detectors are run in cascade, stopping as soon as one succeeds.

┌──────────────────────────────────────────────────────────────────────────┐
│  1. MST Dual Criterion   — primary (fast, exact for clean gaps)          │
│  2. KDE Valley Detection — secondary (robust to sparse bridge cells)     │
│  3. GMM + Gap Validation — fallback (shape-aware, validated spatially)   │
└──────────────────────────────────────────────────────────────────────────┘

DETECTOR 1 — MST Dual Criterion
────────────────────────────────
Zahn (1971): for k tissue portions, the MST has exactly k-1 gap-crossing
edges, which are the k-1 largest MST edges. We require TWO conditions to
declare a gap at the k-1 / k boundary position:

  (a) Ratio test:   e[k-2] / e[k-1] >= tau_ratio   (relative jump)
  (b) Z-score test: z_score(e[k-2]) >= tau_z        (statistical outlier)

Calibration from synthetic experiments:
  Single tissue (any shape): max ratio ≤ 1.35, max z ≤ 5.0
  Two portions, gap ≥ 0.3×R: min ratio ≥ 2.5,  min z ≥ 11.0

Defaults: tau_ratio=2.0, tau_z=6.0 → near-zero false positive rate.

DETECTOR 2 — KDE Valley Detection
────────────────────────────────────
Projects coordinates onto PCA axis 1 (maximum variance direction). Computes
1D KDE and finds density valleys below valley_threshold × peak_density. The
KDE aggregates signal from hundreds of cells, making it robust to individual
bridge cells that break the MST gap chain.

DETECTOR 3 — GMM + Spatial Gap Validation
──────────────────────────────────────────
GMM (original smart_align approach) used as a last resort. After GMM assigns
k clusters, a spatial gap is validated:

  min_between_cluster_dist / median_intra_cluster_nn >= gap_ratio_min

Calibration: single tissue (elongated) max gap_ratio ≤ 2.14,
             two portions (gap ≥ 0.3×R) min gap_ratio ≥ 8.1.

Default gap_ratio_min=3.0 completely separates these populations, preventing
GMM false positives on elongated or concave single-tissue shapes.

REFERENCES
──────────
  Zahn C.T. (1971) Graph-theoretical methods for detecting Gestalt clusters.
      IEEE Trans Comput, 20(1), 68-86.
  Silverman B.W. (1986) Density Estimation for Statistics and Data Analysis.
"""

from __future__ import annotations

import math
import warnings
import numpy as np
import anndata
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Dict


# ─────────────────────────────────────────────────────────────────────────────
# MST utilities
# ─────────────────────────────────────────────────────────────────────────────

def _build_mst(coords: np.ndarray, knn_build: int = 15) -> sp.csr_matrix:
    """Build Euclidean MST. Exact for n≤5000; BallTree-sparse for n>5000."""
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


def _build_components(mst: sp.csr_matrix, k: int, n: int) -> np.ndarray:
    """Remove k-1 largest MST edges → connected component labels."""
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
# Detector 1: MST Dual Criterion
# ─────────────────────────────────────────────────────────────────────────────

def _detect_k_mst_dual(
    edge_weights: np.ndarray,
    max_portions: int = 6,
    tau_ratio: float = 2.0,
    tau_z: float = 6.0,
) -> Tuple[int, float, float]:
    """
    Detect k using a dual criterion: BOTH ratio >= tau_ratio AND z >= tau_z.
    Returns (k, best_boundary_ratio, best_boundary_z).
    """
    sorted_desc = np.sort(edge_weights)[::-1]
    n_check = min(max_portions - 1, len(sorted_desc) - 1)
    if n_check < 1:
        return 1, 0.0, 0.0

    best_k, best_ratio, best_z = 1, 0.0, 0.0
    for m in range(n_check):
        gap_edge = sorted_desc[m]
        remaining = sorted_desc[m + 1:]
        if len(remaining) < 2:
            break
        ratio = gap_edge / (sorted_desc[m + 1] + 1e-12)
        z = (gap_edge - np.mean(remaining)) / (np.std(remaining) + 1e-12)
        if ratio >= tau_ratio and z >= tau_z:
            k = m + 2
            if k > best_k:
                best_k, best_ratio, best_z = k, ratio, z
    return best_k, best_ratio, best_z


# ─────────────────────────────────────────────────────────────────────────────
# Detector 2: KDE Valley Detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_k_kde(
    coords: np.ndarray,
    max_portions: int = 6,
    valley_threshold: float = 0.20,
    n_grid: int = 512,
) -> Tuple[int, np.ndarray]:
    """
    KDE valley detection along PCA axis 1 (maximum variance direction).
    Returns (k, per-cell labels).
    """
    n = len(coords)
    if n < 10:
        return 1, np.zeros(n, dtype=int)

    pca = PCA(n_components=1)
    proj = pca.fit_transform(coords).ravel()

    try:
        kde = gaussian_kde(proj, bw_method='scott')
    except Exception:
        return 1, np.zeros(n, dtype=int)

    x_grid = np.linspace(proj.min(), proj.max(), n_grid)
    density = kde(x_grid)
    peak_density = density.max()
    gap_threshold = valley_threshold * peak_density

    valleys = [
        i for i in range(1, len(density) - 1)
        if density[i] < density[i - 1] and density[i] < density[i + 1]
           and density[i] < gap_threshold
    ]
    if not valleys:
        return 1, np.zeros(n, dtype=int)

    selected = sorted(sorted(valleys, key=lambda i: density[i])[:max_portions - 1])
    k = len(selected) + 1
    labels = np.zeros(n, dtype=int)
    for j, idx in enumerate(selected):
        labels[proj > x_grid[idx]] = j + 1
    return k, labels


# ─────────────────────────────────────────────────────────────────────────────
# Detector 3: GMM + Spatial Gap Validation
# ─────────────────────────────────────────────────────────────────────────────

def _spatial_gap_ratio(coords: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute: min_between_cluster_distance / median_intra_cluster_nn_distance.

    This ratio is < 2.5 for split continuous tissue (no real gap).
    It is > 8 for any two genuinely separate tissue portions.
    Safe threshold: gap_ratio_min = 3.0.
    """
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0

    # Median nearest-neighbour distance within clusters (characteristic cell spacing)
    intra_nns = []
    cluster_pts = {}
    for lbl in unique:
        pts = coords[labels == lbl]
        cluster_pts[lbl] = pts
        if len(pts) >= 2:
            tree = cKDTree(pts)
            dists, _ = tree.query(pts, k=2)
            intra_nns.extend(dists[:, 1].tolist())
    median_intra = float(np.median(intra_nns)) if intra_nns else 1.0

    # Minimum distance between every pair of clusters
    min_gap = float('inf')
    lbls = list(unique)
    for i in range(len(lbls)):
        for j in range(i + 1, len(lbls)):
            tree_j = cKDTree(cluster_pts[lbls[j]])
            dists, _ = tree_j.query(cluster_pts[lbls[i]], k=1)
            min_gap = min(min_gap, float(dists.min()))

    return min_gap / (median_intra + 1e-10)


def _detect_k_gmm_validated(
    coords: np.ndarray,
    max_portions: int = 6,
    silhouette_threshold: float = 0.35,
    min_mass_fraction: float = 0.05,
    gap_ratio_min: float = 3.0,
    n_seeds: int = 3,
) -> Tuple[int, np.ndarray]:
    """
    GMM detection with mandatory spatial gap validation.
    A k>1 result is only accepted if the spatial gap ratio >= gap_ratio_min,
    preventing false positives on elongated or concave single-tissue shapes.
    """
    total = len(coords)
    best_k, best_labels, best_score = 1, np.zeros(total, dtype=int), -1

    for k in range(2, max_portions + 1):
        scores, label_list = [], []
        for seed in range(n_seeds):
            try:
                gmm = GaussianMixture(n_components=k, random_state=seed,
                                      covariance_type='full', n_init=2)
                lbl = gmm.fit_predict(coords)
                score = silhouette_score(coords, lbl)
                _, counts = np.unique(lbl, return_counts=True)
                passes_mass = all(c / total >= min_mass_fraction for c in counts)
                passes_gap  = _spatial_gap_ratio(coords, lbl) >= gap_ratio_min
                if score > silhouette_threshold and passes_mass and passes_gap:
                    scores.append(score)
                    label_list.append(lbl)
            except Exception:
                pass
        if len(scores) >= 2:
            avg = float(np.mean(scores))
            if avg > best_score:
                best_score = avg
                best_k = k
                best_labels = label_list[int(np.argmax(scores))]

    return best_k, best_labels


# ─────────────────────────────────────────────────────────────────────────────
# Fragment handling
# ─────────────────────────────────────────────────────────────────────────────

def _merge_small_fragments(
    labels: np.ndarray, coords: np.ndarray, min_mass_fraction: float
) -> np.ndarray:
    """Merge components below min_mass_fraction into nearest large component."""
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
    edge_weights: np.ndarray, k_ref: int, max_portions: int = 6,
    n_steps: int = 15,
) -> float:
    """Fraction of (tau_ratio, tau_z) pairs in calibrated ranges giving k == k_ref."""
    tau_ratios = np.linspace(1.5, 4.0, n_steps)
    tau_zs     = np.linspace(4.0, 10.0, n_steps)
    agree = sum(
        1 for tr, tz in zip(tau_ratios, tau_zs)
        if _detect_k_mst_dual(edge_weights, max_portions, tr, tz)[0] == k_ref
    )
    return agree / n_steps


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

class PortionDetectionResult:
    """Structured result from find_spatial_portions_mst."""
    def __init__(self, k, labels, detector_used, k_per_detector,
                 gap_threshold, stability_score, debug_info):
        self.k = k
        self.labels = labels
        self.detector_used = detector_used
        self.k_per_detector = k_per_detector
        self.gap_threshold = gap_threshold
        self.stability_score = stability_score
        self.debug_info = debug_info

    def __repr__(self):
        return (f"PortionDetectionResult(k={self.k}, "
                f"detector='{self.detector_used}', "
                f"per_detector={self.k_per_detector})")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def find_spatial_portions_mst(
    adata: anndata.AnnData,
    min_mass_fraction: float = 0.05,
    max_portions: int = 6,
    tau_ratio: float = 2.0,
    tau_z: float = 6.0,
    valley_threshold: float = 0.20,
    silhouette_threshold: float = 0.35,
    gap_ratio_min: float = 3.0,
    use_gmm_fallback: bool = True,
    merge_fragments: bool = True,
    knn_build: int = 15,
    verbose: bool = False,
) -> PortionDetectionResult:
    """
    Detect physically distinct tissue portions using a cascaded ensemble:
    MST Dual → KDE Valley → GMM + Gap Validation.

    Parameters
    ----------
    adata               : AnnData with .obsm['spatial'] (n × 2).
    min_mass_fraction   : Minimum cell fraction per valid portion (default 0.05).
    max_portions        : Biological upper bound on k (default 6).
    tau_ratio           : MST boundary ratio threshold (default 2.0).
                          Handles gaps as tight as 0.3× tissue radius.
    tau_z               : MST z-score threshold (default 6.0).
                          6 SDs above intra-tissue edge distribution.
    valley_threshold    : KDE valley depth threshold (default 0.20).
                          Valley must be <20% of peak density.
    silhouette_threshold: GMM acceptance threshold (default 0.35).
    gap_ratio_min       : Spatial gap validation threshold for GMM (default 3.0).
                          Prevents GMM false positives on elongated tissue.
    use_gmm_fallback    : If False, skip GMM (faster, more conservative).
    merge_fragments     : Merge sub-threshold debris fragments (default True).
    knn_build           : k for BallTree sparse MST (n > 5000 case).
    verbose             : Print per-detector diagnostics.

    Returns
    -------
    PortionDetectionResult with .k, .labels, .detector_used, .k_per_detector.
    """
    coords = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    n = len(coords)
    if n < 3:
        raise ValueError(f"Slice has only {n} cells.")

    # ── Detector 1: MST Dual Criterion ──────────────────────────────────
    mst = _build_mst(coords, knn_build=knn_build)
    edge_weights = mst.tocoo().data.copy()
    sorted_desc = np.sort(edge_weights)[::-1]
    n_top = min(max_portions, len(sorted_desc) - 1)
    top_ratios = sorted_desc[:n_top] / (sorted_desc[1:n_top + 1] + 1e-12)

    k_mst, best_ratio, best_z = _detect_k_mst_dual(
        edge_weights, max_portions, tau_ratio, tau_z
    )
    stability = _compute_stability_score(edge_weights, k_mst, max_portions)
    gap_threshold = float(sorted_desc[k_mst - 1]) if k_mst >= 2 else float('inf')

    debug_info = {'top_ratios': top_ratios, 'boundary_ratio': best_ratio,
                  'boundary_z': best_z, 'mst_k': k_mst}

    if verbose:
        print(f"  [Detector 1: MST dual]  k={k_mst}  "
              f"ratio={best_ratio:.2f}  z={best_z:.2f}")

    if k_mst > 1:
        labels = _build_components(mst, k_mst, n)
        if merge_fragments:
            labels = _merge_small_fragments(labels, coords, min_mass_fraction)
        k_mst = len(np.unique(labels))
        return PortionDetectionResult(
            k=k_mst, labels=labels, detector_used='mst_dual',
            k_per_detector={'mst': k_mst, 'kde': None, 'gmm': None},
            gap_threshold=gap_threshold, stability_score=stability,
            debug_info=debug_info,
        )

    # ── Detector 2: KDE Valley Detection ────────────────────────────────
    k_kde, labels_kde = _detect_k_kde(
        coords, max_portions=max_portions, valley_threshold=valley_threshold
    )
    if merge_fragments and k_kde > 1:
        labels_kde = _merge_small_fragments(labels_kde, coords, min_mass_fraction)
        k_kde = len(np.unique(labels_kde))

    debug_info['kde_k'] = k_kde
    if verbose:
        print(f"  [Detector 2: KDE valley] k={k_kde}  "
              f"valley_threshold={valley_threshold}")

    if k_kde > 1:
        return PortionDetectionResult(
            k=k_kde, labels=labels_kde, detector_used='kde',
            k_per_detector={'mst': k_mst, 'kde': k_kde, 'gmm': None},
            gap_threshold=float('nan'), stability_score=stability,
            debug_info=debug_info,
        )

    # ── Detector 3: GMM + Spatial Gap Validation ─────────────────────────
    if use_gmm_fallback:
        k_gmm, labels_gmm = _detect_k_gmm_validated(
            coords, max_portions=max_portions,
            silhouette_threshold=silhouette_threshold,
            min_mass_fraction=min_mass_fraction,
            gap_ratio_min=gap_ratio_min,
        )
        debug_info['gmm_k'] = k_gmm
        if verbose:
            print(f"  [Detector 3: GMM+gap]   k={k_gmm}  "
                  f"gap_ratio_min={gap_ratio_min}")

        if k_gmm > 1:
            if merge_fragments:
                labels_gmm = _merge_small_fragments(labels_gmm, coords, min_mass_fraction)
                k_gmm = len(np.unique(labels_gmm))
            return PortionDetectionResult(
                k=k_gmm, labels=labels_gmm, detector_used='gmm',
                k_per_detector={'mst': k_mst, 'kde': k_kde, 'gmm': k_gmm},
                gap_threshold=float('nan'), stability_score=stability,
                debug_info=debug_info,
            )
    else:
        debug_info['gmm_k'] = None

    # ── All detectors agree: single portion ─────────────────────────────
    if verbose:
        print(f"  [Ensemble] All detectors agree k=1")

    return PortionDetectionResult(
        k=1, labels=np.zeros(n, dtype=int),
        detector_used='unanimous_k1',
        k_per_detector={'mst': k_mst, 'kde': k_kde,
                        'gmm': debug_info.get('gmm_k')},
        gap_threshold=float('inf'), stability_score=stability,
        debug_info=debug_info,
    )


def find_spatial_portions(
    adata: anndata.AnnData,
    config,
    max_portions: int = 4,
) -> Tuple[int, np.ndarray]:
    """
    Drop-in replacement for find_spatial_portions in smart_align.py.

    Runs the full cascade (MST dual → KDE valley → GMM + gap validation).
    Prints per-detector diagnostics via verbose=True.
    """
    result = find_spatial_portions_mst(
        adata,
        min_mass_fraction=config.min_mass_fraction,
        max_portions=max_portions,
        silhouette_threshold=getattr(config, 'silhouette_threshold', 0.35),
        use_gmm_fallback=True,
        merge_fragments=True,
        verbose=True,
    )
    print(f"  → Ensemble: k={result.k} via '{result.detector_used}'  "
          f"all detectors={result.k_per_detector}")
    return result.k, result.labels


# Backward compatibility alias for sensitivity_analysis.py
def _detect_k_from_mst(edge_weights, max_portions=6, ratio_threshold=2.0):
    k, _, _ = _detect_k_mst_dual(edge_weights, max_portions, ratio_threshold, tau_z=0.0)
    return k