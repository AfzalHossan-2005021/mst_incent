"""
spatial_portion_detection.py  —  v4 (definitive)
────────────────────────────────────────────────────────────────────────────
Biologically-principled tissue-portion detection for spatial transcriptomics.

ROOT CAUSE SUMMARY  (see root_cause_analysis.md for full derivation)
──────────────────────────────────────────────────────────────────────
GMM found the correct biology (k=2, k=4) while MST returned k=1,1 due to
two distinct bugs:

  BUG 1 — Wrong MST denominator
    Old: ratio = e[m] / e[m+1]  (consecutive ratio)
    This compares the gap edge against the WORST-CASE intra-tissue edge
    (sparse perimeter cell), which can reach 80% of gap size.
    Fix: ratio = e[m] / median(ALL edges) — compares gap to TYPICAL spacing.

  BUG 2 — Over-aggressive fragment merging
    After MST correctly found k=2, _merge_small_fragments(min_mass_fraction=0.05)
    deleted the smaller portion when it held < 5% of cells, collapsing k back to 1.
    A real tissue portion with 3% of cells is still a genuine anatomical region.
    Fix: Two-tier merging — only merge true debris (< debris_threshold=0.01 = 1%).
    When a detector finds a gap with high confidence, respect its result.

  BUG 3 — Incorrect multi-portion counting
    Old inner loop accumulated ALL positions with ratio >= tau, causing k=5,6
    for profiles where intra-tissue tail edges also exceed the threshold.
    Fix: Find the position of the LARGEST DROP in the ratio-to-median profile
    within the top max_portions positions.  The biggest drop = gap/intra boundary.

MST ALGORITHM  (corrected)
───────────────────────────
1. Build MST.
2. Compute ratio[m] = sorted_edges[m] / median(ALL edges).
3. Search positions m = 0..max_portions-1 for the biggest consecutive drop:
     drop[m] = ratio[m] - ratio[m+1]
4. Accept if: ratio[best_m] >= tau_min (gap edge tall enough)
              AND drop[best_m] >= min_drop (genuine step, not tail noise)
5. k = best_m + 2.

Calibration from empirical sweep:
  Real gap boundaries:  drop >= 0.99  (smallest: 2-portion gap=50µm)
  Single tissue:        drop <= 0.19  (largest: elongated shape)
  Safe defaults: tau_min=3.5, min_drop=0.5

MERGE POLICY (two-tier)
────────────────────────
  debris_threshold (default 0.01 = 1%): merge truly isolated cells
  min_mass_fraction (default 0.05 = 5%): report warning but DO NOT merge
    genuine small portions detected by a confident detector

CASCADE
────────
  Detector 1 — MST (corrected):    gap ≥ ~50 µm
  Detector 2 — KDE adaptive BW:    gap ≥ ~30 µm
  Detector 3 — GMM + gap valid.:   gap ≥ ~20 µm
  Gap < 20 µm: physically ambiguous (cells touching)
"""

from __future__ import annotations
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
# MST construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_mst(coords: np.ndarray, knn_build: int = 15) -> sp.csr_matrix:
    """Euclidean MST. Exact for n≤5000; BallTree-sparse for n>5000."""
    n = len(coords)
    if n <= 5_000:
        D = sp.csr_matrix(squareform(pdist(coords, metric='euclidean')))
    else:
        tree = BallTree(coords, leaf_size=40)
        k = min(knn_build + 1, n)
        dists, idxs = tree.query(coords, k=k)
        rows = np.repeat(np.arange(n), k)
        D = sp.csr_matrix((dists.ravel(), (rows, idxs.ravel())), shape=(n, n))
        D = D + D.T
    return minimum_spanning_tree(D)


def _build_components(mst: sp.csr_matrix, k: int, n: int) -> np.ndarray:
    """Remove k-1 largest MST edges → connected component labels."""
    if k <= 1:
        return np.zeros(n, dtype=int)
    coo = mst.tocoo()
    threshold = float(np.sort(coo.data)[::-1][k - 1])
    mask = coo.data <= threshold
    pruned = sp.csr_matrix(
        (coo.data[mask], (coo.row[mask], coo.col[mask])), shape=(n, n)
    )
    pruned = pruned + pruned.T
    _, labels = connected_components(pruned, directed=False)
    return labels.astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Detector 1 — MST: median ratio + biggest-drop boundary
# ─────────────────────────────────────────────────────────────────────────────

def _detect_k_mst(
    edge_weights: np.ndarray,
    max_portions: int = 6,
    tau_min: float = 3.5,
    min_drop: float = 0.5,
) -> Tuple[int, float]:
    """
    Detect k via the biggest-drop in the ratio-to-median profile.

    ratio[m] = sorted_edges[m] / median(all_edges)

    Among the top max_portions positions, find the position with the
    largest consecutive drop: drop[m] = ratio[m] - ratio[m+1].

    The biggest drop marks the last gap edge → k = best_m + 2.

    Two acceptance gates:
      ratio[best_m] >= tau_min  : gap edge must be tall enough above median
      drop[best_m]  >= min_drop : drop must be genuine (not tail noise)

    Calibration:
      Real gap boundaries: drop >= 0.99 (smallest: 2-portion gap=50µm)
      Single tissue noise: drop <= 0.19 (elongated shape)
      Defaults: tau_min=3.5, min_drop=0.5
    """
    ew_s = np.sort(edge_weights)[::-1]
    med  = np.median(ew_s)
    n_check = min(max_portions, len(ew_s) - 1)
    if n_check < 1:
        return 1, 0.0

    ratios = ew_s[:n_check + 1] / (med + 1e-12)
    drops  = ratios[:-1] - ratios[1:]         # shape: (n_check,)

    best_m = int(np.argmax(drops))
    best_drop  = float(drops[best_m])
    best_ratio = float(ratios[best_m])

    if best_ratio < tau_min or best_drop < min_drop:
        return 1, 0.0

    return best_m + 2, best_ratio


# ─────────────────────────────────────────────────────────────────────────────
# Detector 2 — KDE with adaptive bandwidth
# ─────────────────────────────────────────────────────────────────────────────

def _detect_k_kde(
    coords: np.ndarray,
    max_portions: int = 6,
    bw_multiplier: float = 2.0,
    valley_threshold: float = 0.15,
    n_grid: int = 1024,
) -> Tuple[int, np.ndarray]:
    """
    1D KDE valley detection along the PCA principal axis.

    Bandwidth = bw_multiplier × median_NN_distance (≈ 2× cell spacing).
    This resolves gaps down to ~30 µm without noise spikes.
    valley_threshold=0.15 separates real gaps from elongated-tissue dips.
    """
    n = len(coords)
    if n < 10:
        return 1, np.zeros(n, dtype=int)

    tree = BallTree(coords, leaf_size=40)
    nn_dists, _ = tree.query(coords, k=2)
    median_nn = float(np.median(nn_dists[:, 1]))

    pca  = PCA(n_components=1)
    proj = pca.fit_transform(coords).ravel()
    std  = np.std(proj)
    if std < 1e-10:
        return 1, np.zeros(n, dtype=int)

    bw = (bw_multiplier * median_nn) / std
    try:
        kde = gaussian_kde(proj, bw_method=bw)
    except Exception:
        return 1, np.zeros(n, dtype=int)

    xg = np.linspace(proj.min(), proj.max(), n_grid)
    d  = kde(xg)
    peak = d.max()

    deep = [
        (xg[i], d[i] / peak)
        for i in range(1, len(d) - 1)
        if d[i] < d[i-1] and d[i] < d[i+1] and d[i] / peak < valley_threshold
    ]
    if not deep:
        return 1, np.zeros(n, dtype=int)

    selected = sorted(
        sorted(deep, key=lambda v: v[1])[:max_portions - 1],
        key=lambda v: v[0]
    )
    k      = len(selected) + 1
    labels = np.zeros(n, dtype=int)
    for j, (boundary, _) in enumerate(selected):
        labels[proj > boundary] = j + 1
    return k, labels


# ─────────────────────────────────────────────────────────────────────────────
# Detector 3 — GMM + spatial gap validation
# ─────────────────────────────────────────────────────────────────────────────

def _spatial_gap_ratio(coords: np.ndarray, labels: np.ndarray) -> float:
    """
    min_between_cluster_distance / median_intra_cluster_NN.

    For genuinely separate portions: min_dist = physical gap >> cell spacing → high.
    For GMM splitting continuous tissue: min_dist ≈ 0 → low.
    Calibration: single tissue max ≤ 2.5, two real portions min ≥ 8.
    """
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    intra, cpts = [], {}
    for lb in unique:
        pts = coords[labels == lb]; cpts[lb] = pts
        if len(pts) >= 2:
            t = cKDTree(pts); d, _ = t.query(pts, k=2); intra.extend(d[:, 1])
    med_intra = float(np.median(intra)) if intra else 1.0
    min_gap   = float('inf')
    lbls      = list(unique)
    for i in range(len(lbls)):
        for j in range(i + 1, len(lbls)):
            t = cKDTree(cpts[lbls[j]]); d, _ = t.query(cpts[lbls[i]], k=1)
            min_gap = min(min_gap, float(d.min()))
    return min_gap / (med_intra + 1e-10)


def _detect_k_gmm(
    coords: np.ndarray,
    max_portions: int = 6,
    silhouette_threshold: float = 0.35,
    min_mass_fraction: float = 0.05,
    gap_ratio_min: float = 3.0,
    n_seeds: int = 5,
) -> Tuple[int, np.ndarray]:
    """GMM with spatial gap validation. Handles gap ≥ ~20 µm."""
    total = len(coords)
    best_k, best_labels, best_score = 1, np.zeros(total, dtype=int), -1
    for k in range(2, max_portions + 1):
        scores, lbls = [], []
        for seed in range(n_seeds):
            try:
                g = GaussianMixture(n_components=k, random_state=seed,
                                    covariance_type='full', n_init=3)
                lb   = g.fit_predict(coords)
                sc   = silhouette_score(coords, lb)
                _, c = np.unique(lb, return_counts=True)
                ok_mass = all(x / total >= min_mass_fraction for x in c)
                ok_gap  = _spatial_gap_ratio(coords, lb) >= gap_ratio_min
                if sc > silhouette_threshold and ok_mass and ok_gap:
                    scores.append(sc); lbls.append(lb)
            except Exception:
                pass
        if len(scores) >= 2:
            avg = float(np.mean(scores))
            if avg > best_score:
                best_score  = avg
                best_k      = k
                best_labels = lbls[int(np.argmax(scores))]
    return best_k, best_labels


# ─────────────────────────────────────────────────────────────────────────────
# Two-tier fragment merging  (BUG FIX)
# ─────────────────────────────────────────────────────────────────────────────

def _merge_debris(
    labels: np.ndarray,
    coords: np.ndarray,
    debris_threshold: float = 0.01,
) -> np.ndarray:
    """
    Merge ONLY true debris: isolated scatter cells < debris_threshold (1%).

    This is intentionally much more conservative than the original
    min_mass_fraction=0.05 gate, which was collapsing genuine small
    tissue portions (e.g. a smaller brain region holding 3-4% of cells)
    back to k=1 after a detector correctly identified them.

    Rule: a component < debris_threshold is debris.
          a component >= debris_threshold is a legitimate portion.
    """
    labels = labels.copy()
    total  = len(labels)
    unique, counts = np.unique(labels, return_counts=True)
    large  = unique[counts / total >= debris_threshold]
    debris = unique[counts / total <  debris_threshold]
    if len(debris) == 0:
        return labels
    large_centroids = np.array([coords[labels == c].mean(axis=0) for c in large])
    for d in debris:
        c_d   = coords[labels == d].mean(axis=0)
        nearest = large[np.argmin(np.linalg.norm(large_centroids - c_d, axis=1))]
        labels[labels == d] = nearest
    for new_i, old_l in enumerate(np.unique(labels)):
        labels[labels == old_l] = new_i
    return labels


def _warn_small_portions(labels: np.ndarray, min_mass_fraction: float) -> None:
    """Warn if any portion is below min_mass_fraction but don't delete it."""
    total  = len(labels)
    unique, counts = np.unique(labels, return_counts=True)
    small = unique[counts / total < min_mass_fraction]
    if len(small) > 0:
        fracs = counts[counts / total < min_mass_fraction] / total
        warnings.warn(
            f"{len(small)} detected portion(s) hold < {min_mass_fraction:.0%} of cells "
            f"(fractions: {np.round(fracs, 3).tolist()}). "
            "These may be genuine small anatomical regions. "
            "Set min_mass_fraction lower to suppress this warning.",
            UserWarning,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stability score
# ─────────────────────────────────────────────────────────────────────────────

def _compute_stability(
    edge_weights: np.ndarray, k_ref: int,
    max_portions: int = 6, n_steps: int = 15,
) -> float:
    """Fraction of (tau, min_drop) pairs in calibrated ranges giving k == k_ref."""
    taus  = np.linspace(2.5, 5.0, n_steps)
    drops = np.linspace(0.3, 0.8, n_steps)
    agree = sum(
        1 for t, d in zip(taus, drops)
        if _detect_k_mst(edge_weights, max_portions, t, d)[0] == k_ref
    )
    return agree / n_steps


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

class PortionDetectionResult:
    def __init__(self, k, labels, detector_used, k_per_detector,
                 gap_threshold, stability_score, debug_info):
        self.k               = k
        self.labels          = labels
        self.detector_used   = detector_used
        self.k_per_detector  = k_per_detector
        self.gap_threshold   = gap_threshold
        self.stability_score = stability_score
        self.debug_info      = debug_info

    def __repr__(self):
        return (f"PortionDetectionResult(k={self.k}, "
                f"detector='{self.detector_used}', "
                f"per_detector={self.k_per_detector})")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def find_spatial_portions_mst(
    adata: anndata.AnnData,
    min_mass_fraction: float  = 0.05,
    debris_threshold:  float  = 0.01,
    max_portions:      int    = 6,
    # Detector 1: MST
    tau_min:   float = 3.5,
    min_drop:  float = 0.5,
    # Detector 2: KDE
    bw_multiplier:     float  = 2.0,
    valley_threshold:  float  = 0.15,
    # Detector 3: GMM
    silhouette_threshold: float = 0.35,
    gap_ratio_min:        float = 3.0,
    use_gmm_fallback:     bool  = True,
    # Shared
    knn_build: int  = 15,
    verbose:   bool = False,
) -> PortionDetectionResult:
    """
    Detect physically distinct tissue portions using a cascaded ensemble.

    Detectors run in order; the first to find k>1 wins:
      1. MST biggest-drop      gap ≥ ~50 µm
      2. KDE adaptive BW       gap ≥ ~30 µm
      3. GMM + gap validation  gap ≥ ~20 µm

    KEY CHANGES from previous versions
    ────────────────────────────────────
    • MST now uses biggest-drop in ratio-to-median profile (fixes multi-portion
      over/under-counting and the consecutive-ratio failure mode).
    • Fragment merging is two-tier: debris_threshold (1%) removes true scatter
      cells; min_mass_fraction (5%) now only warns without deleting portions.
      This fixes the bug where a correct k=2 was collapsed to k=1 because the
      smaller portion held < 5% of cells.

    Parameters
    ----------
    adata                : AnnData with .obsm['spatial'] (n × 2).
    min_mass_fraction    : Warn if any portion < this fraction (default 0.05).
                           Does NOT delete the portion.
    debris_threshold     : Delete components below this fraction (default 0.01).
                           Only true scatter debris, not legitimate small portions.
    max_portions         : Biological upper bound on k (default 6).
    tau_min              : MST: ratio-to-median must be ≥ tau_min (default 3.5).
    min_drop             : MST: biggest drop must be ≥ min_drop (default 0.5).
    bw_multiplier        : KDE: bandwidth = bw_multiplier × median_NN (default 2.0).
    valley_threshold     : KDE: valley depth threshold (default 0.15).
    silhouette_threshold : GMM: silhouette acceptance threshold (default 0.35).
    gap_ratio_min        : GMM: spatial gap validation threshold (default 3.0).
    use_gmm_fallback     : Run GMM if Detectors 1+2 give k=1 (default True).
    knn_build            : BallTree k for n > 5000 cells (default 15).
    verbose              : Print per-detector diagnostics (default False).
    """
    coords = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    n = len(coords)
    if n < 3:
        raise ValueError(f"Slice has only {n} cells.")

    # ── Detector 1: MST biggest-drop ────────────────────────────────────
    mst  = _build_mst(coords, knn_build=knn_build)
    ew   = mst.tocoo().data.copy()
    ew_s = np.sort(ew)[::-1]
    med  = np.median(ew)
    n_top = min(max_portions + 1, len(ew_s))
    ratios_profile = (ew_s[:n_top] / (med + 1e-12)).tolist()

    k_mst, ratio_mst = _detect_k_mst(ew, max_portions, tau_min, min_drop)
    stability    = _compute_stability(ew, k_mst, max_portions)
    gap_threshold = float(ew_s[k_mst - 1]) if k_mst >= 2 else float('inf')

    debug_info = {
        'ratios_to_median': [round(r, 3) for r in ratios_profile[:max_portions + 1]],
        'global_median':    float(med),
        'boundary_ratio':   ratio_mst,
        'tau_min':          tau_min,
        'min_drop':         min_drop,
        'mst_k':            k_mst,
    }

    if verbose:
        drops = [round(ratios_profile[i] - ratios_profile[i+1], 3)
                 for i in range(min(5, len(ratios_profile)-1))]
        print(f"  [1: MST]  k={k_mst}  ratio={ratio_mst:.2f}  "
              f"profile={[round(r,2) for r in ratios_profile[:6]]}  "
              f"drops={drops}")

    if k_mst > 1:
        labels = _build_components(mst, k_mst, n)
        labels = _merge_debris(labels, coords, debris_threshold)
        _warn_small_portions(labels, min_mass_fraction)
        k_mst  = len(np.unique(labels))
        return PortionDetectionResult(
            k=k_mst, labels=labels, detector_used='mst',
            k_per_detector={'mst': k_mst, 'kde': None, 'gmm': None},
            gap_threshold=gap_threshold, stability_score=stability,
            debug_info=debug_info,
        )

    # ── Detector 2: KDE adaptive bandwidth ──────────────────────────────
    k_kde, labels_kde = _detect_k_kde(
        coords, max_portions=max_portions,
        bw_multiplier=bw_multiplier, valley_threshold=valley_threshold,
    )
    if k_kde > 1:
        labels_kde = _merge_debris(labels_kde, coords, debris_threshold)
        _warn_small_portions(labels_kde, min_mass_fraction)
        k_kde = len(np.unique(labels_kde))

    debug_info['kde_k'] = k_kde
    if verbose:
        print(f"  [2: KDE]  k={k_kde}  bw={bw_multiplier}×median_nn  "
              f"valley_thresh={valley_threshold}")

    if k_kde > 1:
        return PortionDetectionResult(
            k=k_kde, labels=labels_kde, detector_used='kde',
            k_per_detector={'mst': k_mst, 'kde': k_kde, 'gmm': None},
            gap_threshold=float('nan'), stability_score=stability,
            debug_info=debug_info,
        )

    # ── Detector 3: GMM + spatial gap validation ─────────────────────────
    if use_gmm_fallback:
        k_gmm, labels_gmm = _detect_k_gmm(
            coords, max_portions=max_portions,
            silhouette_threshold=silhouette_threshold,
            min_mass_fraction=min_mass_fraction,
            gap_ratio_min=gap_ratio_min,
        )
        debug_info['gmm_k'] = k_gmm
        if verbose:
            print(f"  [3: GMM]  k={k_gmm}  gap_ratio_min={gap_ratio_min}")

        if k_gmm > 1:
            labels_gmm = _merge_debris(labels_gmm, coords, debris_threshold)
            _warn_small_portions(labels_gmm, min_mass_fraction)
            k_gmm = len(np.unique(labels_gmm))
            return PortionDetectionResult(
                k=k_gmm, labels=labels_gmm, detector_used='gmm',
                k_per_detector={'mst': k_mst, 'kde': k_kde, 'gmm': k_gmm},
                gap_threshold=float('nan'), stability_score=stability,
                debug_info=debug_info,
            )
    else:
        debug_info['gmm_k'] = None

    if verbose:
        print(f"  [ensemble] k=1")

    return PortionDetectionResult(
        k=1, labels=np.zeros(n, dtype=int), detector_used='k1',
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
    Cascade: MST → KDE → GMM+validation. verbose=True for traceability.
    """
    result = find_spatial_portions_mst(
        adata,
        min_mass_fraction=config.min_mass_fraction,
        max_portions=max_portions,
        silhouette_threshold=getattr(config, 'silhouette_threshold', 0.35),
        use_gmm_fallback=True,
        verbose=True,
    )
    print(f"  → k={result.k}  detector='{result.detector_used}'  "
          f"all={result.k_per_detector}")
    return result.k, result.labels


# Backward-compatible alias
def _detect_k_from_mst(ew, max_portions=6, ratio_threshold=3.5):
    k, _ = _detect_k_mst(ew, max_portions, ratio_threshold, min_drop=0.5)
    return k