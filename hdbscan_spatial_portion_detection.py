"""
hdbscan_spatial_portion_detection.py
────────────────────────────────────────────────────────────────────────────
High-accuracy macroscopic tissue portion detection for Spatial Transcriptomics.

This module implements a robust HDBSCAN-based detector designed specifically
for spatial transcriptomics pipelines such as INCENT.

Key Improvements
────────────────
1. Adaptive spatial scale estimation via kNN statistics
2. Stability filtering using HDBSCAN probabilities
3. Radius-limited noise reassignment to prevent false bridges
4. Spatial connectivity enforcement
5. Vectorized label normalization

References
──────────
Campello et al. (2013) Density-based clustering based on hierarchical density estimates.
"""

import numpy as np
import anndata
import warnings
from typing import Tuple

from sklearn.neighbors import BallTree, NearestNeighbors
from scipy.sparse.csgraph import connected_components

import hdbscan


# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

class HDBSCANPortionConfig:
    """
    Configuration mapping for HDBSCAN parameters based on biological priors.
    """

    def __init__(
        self,
        min_mass_fraction: float = 0.05,
        min_samples_fraction: float = 0.01,
        stability_threshold: float = 0.3,
        knn_k: int = 20,
    ):
        self.min_mass_fraction = min_mass_fraction
        self.min_samples_fraction = min_samples_fraction
        self.stability_threshold = stability_threshold
        self.knn_k = knn_k


# ────────────────────────────────────────────────────────────────────────────
# Adaptive Spatial Scale Estimation
# ────────────────────────────────────────────────────────────────────────────

def estimate_spatial_scale(coords: np.ndarray, k: int = 20) -> float:
    """
    Estimate local spatial scale using k-nearest-neighbor statistics.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    distances, _ = nbrs.kneighbors(coords)

    # median distance to k-th neighbor
    scale = np.median(distances[:, -1])

    return scale


# ────────────────────────────────────────────────────────────────────────────
# Noise Reassignment
# ────────────────────────────────────────────────────────────────────────────

def reassign_noise(
    coords: np.ndarray,
    labels: np.ndarray,
    radius: float
) -> np.ndarray:
    """
    Reattach noise cells to nearest cluster within a safe radius.
    """

    noise_mask = labels == -1

    if not noise_mask.any():
        return labels

    valid_coords = coords[~noise_mask]
    valid_labels = labels[~noise_mask]

    tree = BallTree(valid_coords)

    noise_coords = coords[noise_mask]

    dist, idx = tree.query(noise_coords, k=1)

    valid = dist[:, 0] < radius

    reassigned = labels.copy()

    reassigned_indices = np.where(noise_mask)[0][valid]

    reassigned[reassigned_indices] = valid_labels[idx[valid, 0]]

    return reassigned


# ────────────────────────────────────────────────────────────────────────────
# Spatial Connectivity Enforcement
# ────────────────────────────────────────────────────────────────────────────

def enforce_spatial_connectivity(coords: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Ensure clusters are spatially connected.
    """

    new_labels = labels.copy()
    current_max = labels.max() + 1

    for cluster_id in np.unique(labels):

        mask = labels == cluster_id

        if mask.sum() < 5:
            continue

        sub_coords = coords[mask]

        nbrs = NearestNeighbors(n_neighbors=6).fit(sub_coords)
        graph = nbrs.kneighbors_graph(sub_coords)

        n_comp, comp_labels = connected_components(graph)

        if n_comp > 1:

            indices = np.where(mask)[0]

            for comp in range(n_comp):

                comp_idx = indices[comp_labels == comp]

                new_labels[comp_idx] = current_max

                current_max += 1

    return new_labels


# ────────────────────────────────────────────────────────────────────────────
# Core Detection
# ────────────────────────────────────────────────────────────────────────────

def detect_hdbscan_portions(
    coords: np.ndarray,
    config,
) -> np.ndarray:

    coords = np.asarray(coords, dtype=np.float32)

    n_cells = len(coords)

    min_mass_fraction = getattr(config, 'min_mass_fraction', 0.05)
    min_samples_fraction = getattr(config, 'min_samples_fraction', 0.01)
    knn_k = getattr(config, 'knn_k', 20)
    stability_threshold = getattr(config, 'stability_threshold', 0.3)

    min_cluster_size = max(10, int(n_cells * min_mass_fraction))
    min_samples = max(5, int(n_cells * min_samples_fraction))

    # Estimate spatial scale
    spatial_scale = float(estimate_spatial_scale(coords, int(knn_k)))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric="euclidean",
        cluster_selection_method="eom",
        cluster_selection_epsilon=float(spatial_scale * 1.5)
    )

    labels = clusterer.fit_predict(coords)

    # Stability filtering
    if hasattr(clusterer, "probabilities_"):

        probs = clusterer.probabilities_

        unstable = probs < stability_threshold

        labels[unstable] = -1

    # Radius-limited noise reassignment
    labels = reassign_noise(coords, labels, radius=spatial_scale * 2)

    # Enforce spatial connectivity
    labels = enforce_spatial_connectivity(coords, labels)

    # Edge case: everything noise
    if (labels == -1).all():

        warnings.warn(
            "HDBSCAN classified all spatial points as noise. Falling back to k=1.",
            UserWarning
        )

        return np.zeros(n_cells, dtype=int)

    # Normalize labels to contiguous integers
    unique, final_labels = np.unique(labels, return_inverse=True)

    return final_labels


# ────────────────────────────────────────────────────────────────────────────
# INCENT Pipeline Entry Point
# ────────────────────────────────────────────────────────────────────────────

def find_spatial_portions_hdbscan(
    adata: anndata.AnnData,
    config: HDBSCANPortionConfig,
    max_portions: int = None,
) -> Tuple[int, np.ndarray]:

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)

    if len(coords) < 3:
        raise ValueError(
            f"Slice has only {len(coords)} cells. Cannot perform structural partitioning."
        )

    labels = detect_hdbscan_portions(coords, config)

    k = len(np.unique(labels))

    print(
        f"[HDBSCAN] Discovered {k} distinct spatial macro-structures based on density topology."
    )

    return k, labels