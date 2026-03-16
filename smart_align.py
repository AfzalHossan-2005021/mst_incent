import numpy as np
import pandas as pd
import anndata
from scipy.spatial.distance import cdist
import itertools
import matplotlib.pyplot as plt
from .INCENT import pairwise_align
from .master_spatial_portion_detection import find_spatial_portions_master as find_spatial_portions

class AlignmentConfig:
    """
    Configuration parameters for spatial alignment.
    Allows user-defined biological priors rather than hardcoded heuristics.
    
    Attributes:
        w_gene: Weight of the gene expression cost in the final candidate scoring.
        w_neighbor: Weight of the neighborhood spatial cost in final scoring.
        min_mass_fraction: Minimum proportion of total cells required for a cluster 
                           to be considered a macro-structure rather than debris.
        min_samples_fraction: Minimum proportion of total cells required for a cluster 
                              to be considered a macro-structure rather than debris.
        silhouette_threshold: Minimum ratio of intra-cluster to inter-cluster distance 
                              required to consider a tissue fragment anatomically disjoint.
        allow_reflection: If true, allows Z-axis mirror reflections in rigid alignment.
        allow_scale: If true, uses Generalized Procrustes Analysis for scale normalization.
        clustering_method: Either 'hierarchical' (continuous) or 'gmm' (ellipsoidal).
        max_candidates: Number of top cluster combinations to evaluate geometrically. 
                        Prevents hardcoded truncation of reasonable geometric candidates.
    """
    def __init__(self,
                 w_gene: float = 1.0,
                 w_neighbor: float = 0.5,
                 min_mass_fraction: float = 0.05,
                 min_samples_fraction: float = 0.01,
                 silhouette_threshold: float = 0.1,
                 allow_reflection: bool = False,
                 allow_scale: bool = True,
                 clustering_method: str = 'gmm',
                 max_candidates: int = 3):
        self.w_gene = w_gene
        self.w_neighbor = w_neighbor
        self.min_mass_fraction = min_mass_fraction
        self.min_samples_fraction = min_samples_fraction
        self.silhouette_threshold = silhouette_threshold
        self.allow_reflection = allow_reflection
        self.allow_scale = allow_scale
        self.clustering_method = clustering_method
        self.max_candidates = max_candidates
        
    def calculate_cost(self, final_obj_gene: float, final_obj_neighbor: float) -> float:
        return (self.w_gene * final_obj_gene) + (self.w_neighbor * final_obj_neighbor)


def get_surviving_indices(slice1, slice2):
    """
    Finds the original indices of cells that survive cell-type filtering.
    """
    shared_cell_types = pd.Index(slice1.obs['cell_type_annot']).unique().intersection(pd.Index(slice2.obs['cell_type_annot']).unique())
    survivors_1 = np.where(slice1.obs['cell_type_annot'].isin(shared_cell_types))[0]
    survivors_2 = np.where(slice2.obs['cell_type_annot'].isin(shared_cell_types))[0]
    return survivors_1, survivors_2


def align_coordinates(coords_A, coords_B, allow_reflection=False, allow_scale=True):
    """
    Finds optimal rigid transformation (rotation, translation, & scale) to functionally 
    align a sub-geometry over a broader geometric space. Returns centered and aligned A, 
    along with centered B.
    Uses Generalized Procrustes Analysis via SVD to capture the tech's slide rotations.
    """
    # 1. Standardize by shifting both to origin solely to compute the rotation matrix
    mean_A = coords_A.mean(axis=0)
    mean_B = coords_B.mean(axis=0)
    
    A_c = coords_A - mean_A
    B_c = coords_B - mean_B
    
    # Scale normalization for SVD math stability
    norm_A = np.linalg.norm(A_c) or 1
    norm_B = np.linalg.norm(B_c) or 1
    A_c_norm = A_c / norm_A
    B_c_norm = B_c / norm_B
    
    # 2. SVD to find optimal rotation angle that maps A onto B
    try:
        U, D, Vt = np.linalg.svd(A_c_norm.T @ B_c_norm)
        R = (U @ Vt).T
        
        # Enforce exactly rotation (No mirror flipping the tissue) unless intended
        if not allow_reflection and np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = (U @ Vt).T
            
        # Compute optimal scaling factor
        if allow_scale:
            # Optimal scale s = trace(R^T A^T B) / trace(A^T A)
            scale = np.trace(R.T @ A_c.T @ B_c) / np.trace(A_c.T @ A_c)
        else:
            scale = 1.0
            
        # 3. Apply the rotation and scaling to the *centered* coordinates
        A_aligned_centered = (A_c @ R.T) * scale
        return A_aligned_centered, B_c
    except Exception:
        # If SVD fails due to extreme mathematical degradation, return centered original
        return A_c, B_c


def get_hausdorff_disparity(coords_A, coords_B, percentile=95, allow_reflection=False, allow_scale=True):
    """
    Computes the geometric shape dissimilarity using a Robust Percentile Directed Hausdorff.
    This effectively ignores stray "floater" cells and experimental debris artifacts 
    that heavily skew standard maximum distance calculations.
    """
    rng = np.random.RandomState(42)
    
    # Safely increased to 3000 since vectorization is highly optimized
    if len(coords_A) > 3000:
        idx_A = rng.choice(len(coords_A), 3000, replace=False)
        cA = coords_A[idx_A]
    else:
        cA = coords_A
        
    if len(coords_B) > 3000:
        idx_B = rng.choice(len(coords_B), 3000, replace=False)
        cB = coords_B[idx_B]
    else:
        cB = coords_B
        
    # Standardize & Rotate
    cA_aligned, cB_centered = align_coordinates(cA, cB, allow_reflection=allow_reflection, allow_scale=allow_scale)
    
    # Calculate pairwise distance matrices
    dist_matrix = cdist(cA_aligned, cB_centered)
    
    # Distance from each point in A to the closest in B, and vice versa
    dists_A_to_B = np.min(dist_matrix, axis=1)
    dists_B_to_A = np.min(dist_matrix, axis=0) # cdist(A,B) axis=0 acts as dist B to A
    
    # Take the robust 95th percentile instead of the 100th (maximum) to ignore extreme outliers/debris
    h1 = np.percentile(dists_A_to_B, percentile)
    h2 = np.percentile(dists_B_to_A, percentile)
    
    return max(h1, h2)


def _visualize_portions_inline(adata, labels, title):
    """
    Inline utility to plot spatial portions discovered during the Smart Align phase.
    """
    coords = adata.obsm['spatial']
    x, y = coords[:, 0], coords[:, 1]
    
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap('tab10' if len(unique_labels) <= 10 else 'tab20')
    
    plt.figure(figsize=(6, 6))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        plt.scatter(x[mask], y[mask], label=f'Portion {lbl}', 
                    color=cmap(i % cmap.N), s=10, alpha=0.9, edgecolors='none')
        
    plt.title(title)
    plt.xlabel('Spatial X')
    plt.ylabel('Spatial Y')
    plt.gca().set_aspect('equal', 'datalim')
    plt.gca().invert_yaxis()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def smart_pairwise_align(sliceA, sliceB, config: AlignmentConfig = None, **kwargs):
    """
    Automatically detects varying numbers of structural portions (e.g., matching a 1-portion slice 
    against a 4-portion heart slice) and perfectly aligns the smaller slice to the correct 
    geographic subset of the larger slice.
    """
    
    # 1. Detect structures dynamically based on spatial density
    if config is None:
        config = AlignmentConfig()
    
    # Import the powerful MST-based structural detection we just built
    k_A, labels_A = find_spatial_portions(sliceA, config)
    k_B, labels_B = find_spatial_portions(sliceB, config)
    
    print(f"[Smart Align] Slice A portions: {k_A} | Slice B portions: {k_B}")
    
    # --- VISUALIZATION INJECTION ---
    _visualize_portions_inline(sliceA, labels_A, f"Slice A Structural Portions (k={k_A})")
    _visualize_portions_inline(sliceB, labels_B, f"Slice B Structural Portions (k={k_B})")
    # -------------------------------
    
    original_return_obj = kwargs.get('return_obj', False)
    kwargs['return_obj'] = True
    
    # Case 1: Slice A has fewer portions than Slice B
    if k_A < k_B:
        print(f"[Smart Align] Slice A ({k_A} portion) is smaller than Slice B ({k_B} portions). Finding best matching sub-geometry...")
        
        # 1. Pre-filter using Directed Hausdorff geometric alignment
        combos_and_disparities = []
        for combo in itertools.combinations(range(k_B), k_A):
            idx_B_combo = np.where(np.isin(labels_B, combo))[0]
            sliceB_sub = sliceB[idx_B_combo]
            
            disparity = get_hausdorff_disparity(
                sliceA.obsm['spatial'], 
                sliceB_sub.obsm['spatial'], 
                allow_reflection=getattr(config, 'allow_reflection', False),
                allow_scale=getattr(config, 'allow_scale', True)
            )
            combos_and_disparities.append((combo, idx_B_combo, disparity))
            
        # Sort by geometry disparity and keep top N maximum to save compute time
        combos_and_disparities.sort(key=lambda x: x[2])
        top_combos = combos_and_disparities[:config.max_candidates]
        
        best_cost = float('inf')
        best_res = None
        best_idx_B = None
        
        # 2. Run Heavy INCENT alignment on the top geometric candidates
        for combo, idx_B_combo, _ in top_combos:
            sliceB_sub = sliceB[idx_B_combo].copy()
            
            iter_kwargs = kwargs.copy()
            iter_kwargs['sliceB_name'] = iter_kwargs.get('sliceB_name', 'B') + f"_parts{combo}"
            res = pairwise_align(sliceA, sliceB_sub, **iter_kwargs)
            
            # Combine Gene Cost + Neighborhood Dist -> Robust Score
            cost = config.calculate_cost(res[4], res[3])
            
            if cost < best_cost:
                best_cost = cost
                best_res = res
                best_idx_B = idx_B_combo
        
        print(f"[Smart Align] Chose Slice B portions mapping to combo (Score: {best_cost:.4f})")
        
        # Reconstruct full Pi matrix into original global dimensions
        surv_A_best, surv_B_sub_best = get_surviving_indices(sliceA, sliceB[best_idx_B])
        full_pi = np.zeros((sliceA.shape[0], sliceB.shape[0]))
        full_pi[np.ix_(surv_A_best, best_idx_B[surv_B_sub_best])] = best_res[0]
        
        best_res_list = list(best_res)
        best_res_list[0] = full_pi

    # Case 2: Slice A has more portions than Slice B
    elif k_A > k_B:
        print(f"[Smart Align] Slice B ({k_B} portions) is smaller than Slice A ({k_A} portions). Finding best matching sub-geometry...")
        
        # 1. Pre-filter using Directed Hausdorff geometric alignment
        combos_and_disparities = []
        for combo in itertools.combinations(range(k_A), k_B):
            idx_A_combo = np.where(np.isin(labels_A, combo))[0]
            sliceA_sub = sliceA[idx_A_combo]
            
            disparity = get_hausdorff_disparity(
                sliceA_sub.obsm['spatial'], 
                sliceB.obsm['spatial'], 
                allow_reflection=getattr(config, 'allow_reflection', False),
                allow_scale=getattr(config, 'allow_scale', True)
            )
            combos_and_disparities.append((combo, idx_A_combo, disparity))
            
        # Sort by geometry disparity and keep top N maximum to save compute time
        combos_and_disparities.sort(key=lambda x: x[2])
        top_combos = combos_and_disparities[:config.max_candidates]
        
        best_cost = float('inf')
        best_res = None
        best_idx_A = None
        
        # 2. Run Heavy INCENT alignment on the top geometric candidates
        for combo, idx_A_combo, _ in top_combos:
            sliceA_sub = sliceA[idx_A_combo].copy()
            
            iter_kwargs = kwargs.copy()
            iter_kwargs['sliceA_name'] = iter_kwargs.get('sliceA_name', 'A') + f"_parts{combo}"
            res = pairwise_align(sliceA_sub, sliceB, **iter_kwargs)
            
            # Combine Gene Cost + Neighborhood Dist -> Robust Score
            cost = config.calculate_cost(res[4], res[3])
            
            if cost < best_cost:
                best_cost = cost
                best_res = res
                best_idx_A = idx_A_combo
        
        print(f"[Smart Align] Chose Slice A portions mapping to combo (Score: {best_cost:.4f})")
        
        # Reconstruct full Pi matrix into original global dimensions
        surv_A_sub_best, surv_B_best = get_surviving_indices(sliceA[best_idx_A], sliceB)
        full_pi = np.zeros((sliceA.shape[0], sliceB.shape[0]))
        full_pi[np.ix_(best_idx_A[surv_A_sub_best], surv_B_best)] = best_res[0]
        
        best_res_list = list(best_res)
        best_res_list[0] = full_pi

    # Case 3: Both have the same number of portions
    else:
        print(f"[Smart Align] Slices have identical number of portions ({k_A} vs {k_B}). Checking shape disparity...")
        disparity = get_hausdorff_disparity(
            sliceA.obsm['spatial'], 
            sliceB.obsm['spatial'], 
            allow_reflection=getattr(config, 'allow_reflection', False),
            allow_scale=getattr(config, 'allow_scale', True)
        )
        print(f"[Smart Align] Full slice structural geometric disparity: {disparity:.4f}. Proceeding with standard alignment.")
        
        surv_A, surv_B = get_surviving_indices(sliceA, sliceB)
        best_res_list = list(pairwise_align(sliceA, sliceB, **kwargs))
        
        # Protect against cell type drop shape mismatches
        full_pi = np.zeros((sliceA.shape[0], sliceB.shape[0]))
        full_pi[np.ix_(surv_A, surv_B)] = best_res_list[0]
        best_res_list[0] = full_pi

    # Obey the user's original request for return objects
    if not original_return_obj:
        return best_res_list[0] # Just the pi matrix
    return tuple(best_res_list)
