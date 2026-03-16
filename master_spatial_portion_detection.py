"""
master_spatial_portion_detection.py
-----------------------------------
State-of-the-Art (SOTA) Spatial Portion Detection for INCENT.
Pipeline:
1. Auto k-NN scale estimation (Elbow method)
2. Bridge edge detection (Edge Betweenness)
3. EMST (Minimum Spanning Tree) coarse segmentation
4. HDBSCAN boundary and density refinement
"""

import warnings
import numpy as np
import anndata
import networkx as nx
import matplotlib.pyplot as plt

from typing import Tuple
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.neighbors import NearestNeighbors
import hdbscan

class MasterPortionConfig:
    def __init__(
        self,
        k_min: int = 3,
        k_max: int = 30,
        bridge_quantile: float = 0.98,
        emst_quantile: float = 0.98,
        min_mass_fraction: float = 0.03,
        plot_diagnostics: bool = False
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.bridge_quantile = bridge_quantile
        self.emst_quantile = emst_quantile
        self.min_mass_fraction = min_mass_fraction
        self.plot_diagnostics = plot_diagnostics


def estimate_optimal_knn(coords: np.ndarray, config) -> int:
    """Step 1: Automatically find the optimal spatial neighborhood scale (k)."""
    k_min = getattr(config, 'k_min', 3)
    k_max = getattr(config, 'k_max', 30)
    plot_diagnostics = getattr(config, 'plot_diagnostics', False)

    k_values = list(range(k_min, k_max + 1))
    avg_distances = []

    for k in k_values:
        nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
        distances, _ = nbrs.kneighbors(coords)
        avg_distances.append(np.mean(distances[:, -1]))

    distances = np.array(avg_distances)
    
    # Second derivative to find the internal elbow stability point
    second_derivative = np.gradient(np.gradient(distances))
    k_opt = k_values[np.argmax(second_derivative)]

    if plot_diagnostics:
        plt.figure(figsize=(5,3))
        plt.plot(k_values, distances, marker="o", label="kNN Dist")
        plt.axvline(k_opt, color="red", linestyle="--", label=f"Optimal k={k_opt}")
        plt.title("Auto kNN Estimation")
        plt.legend()
        plt.show()

    return k_opt


def build_and_prune_graph(coords: np.ndarray, k: int, config):
    """Step 2: Build kNN graph and remove artificial debris bridges."""
    bridge_quantile = getattr(config, 'bridge_quantile', 0.98)

    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    G = nx.Graph()
    n = len(coords)
    
    for i in range(n):
        for j, dist in zip(indices[i], distances[i]):
            if i != j:
                G.add_edge(i, j, weight=dist)

    # Edge Betweenness to detect bottlenecks (bridges) between dense tissues
    bet = nx.edge_betweenness_centrality(G, weight='weight')
    threshold = np.quantile(list(bet.values()), bridge_quantile)
    
    bridge_edges = [edge for edge, val in bet.items() if val > threshold]
    G.remove_edges_from(bridge_edges)
    
    # Convert pruned NetworkX graph to SciPy CSR for EMST
    edges = list(G.edges(data=True))
    if not edges:
        return csr_matrix((n, n))
        
    rows, cols, data = zip(*[(u, v, d['weight']) for u, v, d in edges])
    # Make symmetric
    rows = list(rows) + list(cols)
    cols = list(cols) + list(rows[:len(edges)])
    data = list(data) + list(data)
    
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def coarse_emst_split(graph: csr_matrix, config) -> np.ndarray:
    """Step 3: EMST to partition disconnected sub-graphs after bridge removal."""
    emst_quantile = getattr(config, 'emst_quantile', 0.98)

    mst = minimum_spanning_tree(graph).tocsr()
    
    if mst.nnz > 0:
        threshold = np.quantile(mst.data, emst_quantile)
        mst.data[mst.data > threshold] = 0
        mst.eliminate_zeros()
        
    _, labels = connected_components(mst, directed=False)
    return labels


def refine_with_hdbscan(coords: np.ndarray, coarse_labels: np.ndarray, config) -> np.ndarray:
    """Step 4: Refine boundaries & reject noise inside each EMST portion."""
    min_mass_fraction = getattr(config, 'min_mass_fraction', 0.03)

    final_labels = -np.ones(len(coords), dtype=int)
    label_counter = 0

    for portion in np.unique(coarse_labels):
        mask = coarse_labels == portion
        sub_coords = coords[mask]
        n_cells = len(sub_coords)

        # Skip HDBSCAN for tiny debris islands
        if n_cells < 20:
            final_labels[mask] = -1 # mark as noise
            continue

        min_cluster_size = max(5, int(n_cells * min_mass_fraction))
        min_samples = max(3, int(min_cluster_size * 0.5))

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=int(min_samples),
            metric="euclidean"
        )
        
        sub_labels = clusterer.fit_predict(sub_coords)

        # Remap valid sub-clusters to global distinct labels
        for u in np.unique(sub_labels):
            if u == -1:
                continue # leave as global noise (-1)
            
            submask = (sub_labels == u)
            indices = np.where(mask)[0][submask]
            final_labels[indices] = label_counter
            label_counter += 1

    return final_labels


def find_spatial_portions_master(
    adata: anndata.AnnData,
    config = None,
    max_portions: int = None
) -> Tuple[int, np.ndarray]:
    """
    Master INCENT Entry Point.
    Executes the SOTA hybrid geometry + density framework.
    """
    if config is None:
        config = MasterPortionConfig()

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)

    if len(coords) < 3:
        raise ValueError("Too few spatial points for portion detection.")

    print("[SOTA-Pipeline] 1. Estimating optimal spatial scale...")
    k_opt = estimate_optimal_knn(coords, config)
    print(f"                => Detected optimal k = {k_opt}")

    print("[SOTA-Pipeline] 2. Identifying and pruning spatial bridges...")
    graph = build_and_prune_graph(coords, k_opt, config)

    print("[SOTA-Pipeline] 3. Extracting EMST topology...")
    coarse_labels = coarse_emst_split(graph, config)
    
    print("[SOTA-Pipeline] 4. Refining boundaries with HDBSCAN density...")
    final_labels = refine_with_hdbscan(coords, coarse_labels, config)
    
    # Normalize labels sequentially 
    valid_mask = final_labels != -1
    if np.any(valid_mask):
        unique_valid, final_labels[valid_mask] = np.unique(final_labels[valid_mask], return_inverse=True)
        k = len(unique_valid)
    else:
        # Edge case: everything is noise
        warnings.warn("All spatial points classified as noise. Falling back to k=1.", UserWarning)
        final_labels[:] = 0
        k = 1

    noise_count = np.sum(final_labels == -1)

    print(f"[SOTA-Pipeline] Completed! Detected {k} pure tissue manifolds.")
    if noise_count > 0:
        print(f"                => Filtered {noise_count} noise/debris spots.")

    return k, final_labels
