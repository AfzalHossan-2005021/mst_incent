import numpy as np
import anndata
from typing import Tuple

from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

import hdbscan


class HybridPortionConfig:

    def __init__(
        self,
        emst_quantile: float = 0.98,
        min_mass_fraction: float = 0.03,
        min_samples_fraction: float = 0.01
    ):
        self.emst_quantile = emst_quantile
        self.min_mass_fraction = min_mass_fraction
        self.min_samples_fraction = min_samples_fraction


def build_delaunay_graph(coords):

    tri = Delaunay(coords)

    edges = set()

    for simplex in tri.simplices:

        for i in range(3):
            for j in range(i + 1, 3):

                a = simplex[i]
                b = simplex[j]

                edges.add((a, b))
                edges.add((b, a))

    rows = []
    cols = []
    data = []

    for a, b in edges:

        dist = np.linalg.norm(coords[a] - coords[b])

        rows.append(a)
        cols.append(b)
        data.append(dist)

    n = len(coords)

    graph = csr_matrix((data, (rows, cols)), shape=(n, n))

    return graph


def coarse_emst_split(coords, config):

    graph = build_delaunay_graph(coords)

    mst = minimum_spanning_tree(graph).tocsr()

    edges = mst.data

    threshold = np.quantile(edges, config.emst_quantile)

    mst.data[mst.data > threshold] = 0

    mst.eliminate_zeros()

    n_comp, labels = connected_components(mst)

    return labels


def refine_with_hdbscan(coords, coarse_labels, config):

    final_labels = np.zeros(len(coords), dtype=int)

    label_counter = 0

    for portion in np.unique(coarse_labels):

        mask = coarse_labels == portion

        sub_coords = coords[mask]

        n_cells = len(sub_coords)

        if n_cells < 50:
            # For small scattered fragments, just label as noise (-1) to not over-count
            final_labels[mask] = -1
            continue

        # Use global config size instead of scaling down for sub-portions, 
        # to avoid detecting micro-clusters within already split portions.
        global_n_cells = len(coords)
        min_cluster_size = max(30, int(global_n_cells * config.min_mass_fraction))
        min_samples = max(10, int(global_n_cells * config.min_samples_fraction))

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean"
        )

        sub_labels = clusterer.fit_predict(sub_coords)

        unique = np.unique(sub_labels)

        for u in unique:
            if u == -1:
                # noise stays noise
                submask = sub_labels == u
                indices = np.where(mask)[0][submask]
                final_labels[indices] = -1
                continue

            submask = sub_labels == u

            indices = np.where(mask)[0][submask]

            final_labels[indices] = label_counter

            label_counter += 1

    return final_labels


def detect_hybrid_portions(coords, config):

    coarse_labels = coarse_emst_split(coords, config)

    final_labels = refine_with_hdbscan(coords, coarse_labels, config)
    
    # only count non-noise labels
    valid_mask = final_labels != -1
    if valid_mask.any():
        unique, final_labels[valid_mask] = np.unique(final_labels[valid_mask], return_inverse=True)
    else:
        final_labels[:] = 0

    return final_labels


def find_spatial_portions_hybrid(
    adata: anndata.AnnData,
    config: HybridPortionConfig,
    max_portions: int = None
) -> Tuple[int, np.ndarray]:

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)

    if len(coords) < 3:
        raise ValueError("Too few spatial points for portion detection.")

    labels = detect_hybrid_portions(coords, config)

    k = len(np.unique(labels))

    print(f"[HYBRID] Detected {k} spatial tissue portions.")

    return k, labels