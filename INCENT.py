import os
import ot
import time
import torch
import datetime
import numpy as np
import pandas as pd

from tqdm import tqdm
from anndata import AnnData
from numpy.typing import NDArray
from typing import Optional, Tuple, Union

from .geodesic_topology import compute_geodesic_cost_matrix
from .utils import fused_gromov_wasserstein_incent, to_dense_array, extract_data_matrix, jensenshannon_divergence_backend, pairwise_msd


def pairwise_align(
    sliceA: AnnData, 
    sliceB: AnnData, 
    alpha: float,
    beta: float,
    gamma: float,
    radius: float,
    filePath: str,
    use_rep: Optional[str] = None, 
    G_init = None, 
    a_distribution = None, 
    b_distribution = None, 
    norm: bool = False, 
    numItermax: int = 6000, 
    backend = ot.backend.TorchBackend(), 
    use_gpu: bool = False, 
    return_obj: bool = False,
    verbose: bool = False, 
    gpu_verbose: bool = True, 
    sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None,
    overwrite = False,
    neighborhood_dissimilarity: str='jsd',
    **kwargs) -> Union[NDArray[np.floating], Tuple[NDArray[np.floating], float, float, float, float]]:
    """

    This method is written by Anup Bhowmik, CSE, BUET

    Calculates and returns optimal alignment of two slices of single cell MERFISH data. 
    
    Args:
        sliceA: Slice A to align.
        sliceB: Slice B to align.
        alpha: weight for spatial distance
        beta: weight for cell type one-hot encoding cost
        gamma: weight for neighborhood expression distance (e.g., JSD)
        radius: spatial radius (Euclidean distance) defining the local neighborhood of a cell.
        filePath: Absolute or relative directory path used for caching distance matrices and results.
        use_rep: If ``None``, uses ``slice.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``slice.obsm[use_rep]``.
        G_init (array-like, optional): Initial mapping to be used in FGW-OT, otherwise default is uniform mapping.
        a_distribution (array-like, optional): Distribution of sliceA spots, otherwise default is uniform.
        b_distribution (array-like, optional): Distribution of sliceB spots, otherwise default is uniform.
        norm: If ``True``, scales spatial distances such that neighboring spots are at distance 1. Otherwise, spatial distances remain unchanged.
        numItermax: Max number of iterations during FGW-OT.
        backend: Type of backend to run calculations. For list of backends available on system: ``ot.backend.get_backend_list()``.
        use_gpu: If ``True``, use gpu. Otherwise, use cpu. Currently we only have gpu support for Pytorch.
        return_obj: If ``True``, additionally returns objective function output of FGW-OT.
        verbose: If ``True``, FGW-OT is verbose.
        gpu_verbose: If ``True``, print whether gpu is being used to user.
        sliceA_name: Optional string identifier for slice A caching.
        sliceB_name: Optional string identifier for slice B caching.
        overwrite: If ``True``, forces recalculation of distance matrices ignoring cache.
        neighborhood_dissimilarity: Name of measure for neighborhood comparisons (e.g., ``'jsd'`` for Jensen-Shannon Divergence).

    Returns:
        - Alignment of spots.

        If ``return_obj = True``, additionally returns:
        
        - Objective function output of cost 
    """

    start_time = time.time()

    if not os.path.exists(filePath):
        os.makedirs(filePath)

    logFile = open(f"{filePath}/log.txt", "w")

    logFile.write(f"pairwise_align_INCENT\n")
    currDateTime = datetime.datetime.now()

    # logFile.write(f"{currDateTime.date()}, {currDateTime.strftime("%I:%M %p")} BDT, {currDateTime.strftime("%A")} \n")

    logFile.write(f"{currDateTime}\n")
    logFile.write(f"sliceA_name: {sliceA_name}, sliceB_name: {sliceB_name}\n")
   

    logFile.write(f"alpha: {alpha}\n")
    logFile.write(f"beta: {beta}\n")
    logFile.write(f"gamma: {gamma}\n")
    logFile.write(f"radius: {radius}\n")


    
    # Determine if gpu or cpu is being used
    if use_gpu:
        if torch.cuda.is_available():
            backend = ot.backend.TorchBackend()
            if gpu_verbose:
                print("GPU is requested and available, using gpu.")
        else:
            use_gpu = False
            backend = ot.backend.NumpyBackend()
            if gpu_verbose:
                print("GPU is requested but not available, resorting to torch cpu.")
    else:
        backend = ot.backend.NumpyBackend()
        if gpu_verbose:
            print("Using selected backend cpu. If you want to use gpu, set use_gpu = True.")
    
    # check if slices are valid
    for s in [sliceA, sliceB]:
        if not len(s):
            raise ValueError(f"Found empty `AnnData`:\n{s}.")

    
    # Backend
    nx = backend

    # Filter to shared genes
    shared_genes = sliceA.var_names.intersection(sliceB.var_names)
    if len(shared_genes) == 0:
        raise ValueError("No shared genes between the two slices.")
    sliceA = sliceA[:, shared_genes]
    sliceB = sliceB[:, shared_genes]


    # Filter to shared cell types
    # This is needed for the cell-type mismatch penalty, and also ensures that the neighborhood distributions are comparable (same set of cell types).
    shared_cell_types = pd.Index(sliceA.obs['cell_type_annot']).unique().intersection(pd.Index(sliceB.obs['cell_type_annot']).unique())
    if len(shared_cell_types) == 0:
        raise ValueError("No shared cell types between the two slices.")
    sliceA = sliceA[sliceA.obs['cell_type_annot'].isin(shared_cell_types)]
    sliceB = sliceB[sliceB.obs['cell_type_annot'].isin(shared_cell_types)]

    
    # Calculate spatial distances
    coordinatesA = sliceA.obsm['spatial'].copy()
    coordinatesB = sliceB.obsm['spatial'].copy()
    coordinatesA = nx.from_numpy(coordinatesA)
    coordinatesB = nx.from_numpy(coordinatesB)
    
    if isinstance(nx,ot.backend.TorchBackend):
        coordinatesA = coordinatesA.float()
        coordinatesB = coordinatesB.float()
    D_A = ot.dist(coordinatesA, coordinatesA, metric='euclidean')
    D_B = ot.dist(coordinatesB, coordinatesB, metric='euclidean')

    # --- CRITICAL GEOMETRIC SCALING ---
    # To achieve perfect structural alignment regardless of slices being from different 
    # platforms, resolutions, or mechanical stretch, we normalize the spatial spaces to [0, 1].
    # This ensures Gromov-Wasserstein penalty relies purely on relative shape, not absolute size.
    D_A /= nx.max(D_A)
    D_B /= nx.max(D_B)

    # print the shape of D_A and D_B
    # print("D_A.shape: ", D_A.shape)
    # print("D_B.shape: ", D_B.shape)

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        D_A = D_A.cuda()
        D_B = D_B.cuda()


    # Calculate gene expression dissimilarity
    # filePath = '/content/drive/MyDrive/Thesis_data_anup/local_data'
    cosine_dist_gene_expr = cosine_distance(sliceA, sliceB, sliceA_name, sliceB_name, filePath, use_rep = use_rep, use_gpu = use_gpu, nx = nx, beta = beta, overwrite=overwrite)

    # ── Explicit cell-type mismatch penalty ──────────────────────────────
    # Binary matrix: 0 for same type, 1 for different type.
    # Added to M1 so it enters the FW gradient directly → strong cell-type signal.

    _lab_A = np.asarray(sliceA.obs['cell_type_annot'].values)
    _lab_B = np.asarray(sliceB.obs['cell_type_annot'].values)
    M_celltype = (_lab_A[:, None] != _lab_B[None, :]).astype(np.float64)

    if isinstance(cosine_dist_gene_expr, torch.Tensor):
        M_celltype_t = torch.from_numpy(M_celltype).to(cosine_dist_gene_expr.device)
        M1 = (1 - beta) * cosine_dist_gene_expr + beta * M_celltype_t
    else:
        M1_combined = (1 - beta) * cosine_dist_gene_expr + beta * M_celltype
        M1 = nx.from_numpy(M1_combined)

    logFile.write(f"[cell_type_penalty] beta={beta}, M_celltype shape={M_celltype.shape}\n")


    # jensenshannon_divergence_backend actually returns jensen shannon distance
    # neighborhood_distribution_slice_1, neighborhood_distribution_slice_1 will be pre computed

    if os.path.exists(f"{filePath}/neighborhood_distribution_{sliceA_name}.npy") and not overwrite:
        print("Loading precomputed neighborhood distribution of slice A")
        neighborhood_distribution_sliceA = np.load(f"{filePath}/neighborhood_distribution_{sliceA_name}.npy")
    else:
        print("Calculating neighborhood distribution of slice A")
        neighborhood_distribution_sliceA = neighborhood_distribution(sliceA, radius = radius)


        neighborhood_distribution_sliceA += 0.01 # for avoiding zero division error
        # print("Saving neighborhood distribution of slice A")
        # np.save(f"{filePath}/neighborhood_distribution_{sliceA_name}.npy", neighborhood_distribution_sliceA)


    if os.path.exists(f"{filePath}/neighborhood_distribution_{sliceB_name}.npy") and not overwrite:
        print("Loading precomputed neighborhood distribution of slice B")
        neighborhood_distribution_sliceB = np.load(f"{filePath}/neighborhood_distribution_{sliceB_name}.npy")
    else:
        print("Calculating neighborhood distribution of slice B")
        neighborhood_distribution_sliceB = neighborhood_distribution(sliceB, radius = radius)


        neighborhood_distribution_sliceB += 0.01 # for avoiding zero division error
        # print("Saving neighborhood distribution of slice B")
        # np.save(f"{filePath}/neighborhood_distribution_{sliceB_name}.npy", neighborhood_distribution_sliceB)


    if ('numpy' in str(type(neighborhood_distribution_sliceA))) and use_gpu:
        neighborhood_distribution_sliceA = torch.from_numpy(neighborhood_distribution_sliceA)
    if ('numpy' in str(type(neighborhood_distribution_sliceB))) and use_gpu:
        neighborhood_distribution_sliceB = torch.from_numpy(neighborhood_distribution_sliceB)

    if use_gpu:
        neighborhood_distribution_sliceA = neighborhood_distribution_sliceA.cuda()
        neighborhood_distribution_sliceB = neighborhood_distribution_sliceB.cuda()

    if neighborhood_dissimilarity == 'jsd':
        if os.path.exists(f"{filePath}/js_dist_neighborhood_{sliceA_name}_{sliceB_name}.npy") and not overwrite:
            print("Loading precomputed JSD of neighborhood distribution for slice A and slice B")
            js_dist_neighborhood = np.load(f"{filePath}/js_dist_neighborhood_{sliceA_name}_{sliceB_name}.npy")
            if use_gpu and isinstance(nx, ot.backend.TorchBackend):
                js_dist_neighborhood = torch.from_numpy(js_dist_neighborhood).cuda()
        else:
            print("Calculating JSD of neighborhood distribution for slice A and slice B")

            js_dist_neighborhood = jensenshannon_divergence_backend(neighborhood_distribution_sliceA, neighborhood_distribution_sliceB)

  
        if isinstance(js_dist_neighborhood, torch.Tensor):
            M2 = js_dist_neighborhood
            if use_gpu and js_dist_neighborhood.device.type != 'cuda':
                M2 = M2.cuda()
        else:
            M2 = nx.from_numpy(js_dist_neighborhood)

    elif neighborhood_dissimilarity == 'cosine':
        if isinstance(neighborhood_distribution_sliceA, torch.Tensor) or isinstance(neighborhood_distribution_sliceB, torch.Tensor):
            ndA = neighborhood_distribution_sliceA
            ndB = neighborhood_distribution_sliceB
            if not isinstance(ndA, torch.Tensor):
                ndA = torch.from_numpy(np.asarray(ndA))
            if not isinstance(ndB, torch.Tensor):
                ndB = torch.from_numpy(np.asarray(ndB))
            if use_gpu:
                ndA = ndA.cuda()
                ndB = ndB.cuda()
            numerator = ndA @ ndB.T
            denom = ndA.norm(dim=1)[:, None] * ndB.norm(dim=1)[None, :]
            cosine_dist_neighborhood = 1 - numerator / denom
            M2 = cosine_dist_neighborhood
        else:
            ndA = np.asarray(neighborhood_distribution_sliceA)
            ndB = np.asarray(neighborhood_distribution_sliceB)
            numerator = ndA @ ndB.T
            denom = np.linalg.norm(ndA, axis=1)[:, None] * np.linalg.norm(ndB, axis=1)[None, :]
            cosine_dist_neighborhood = 1 - numerator / denom
            M2 = nx.from_numpy(cosine_dist_neighborhood)

    elif neighborhood_dissimilarity == 'msd':
        if isinstance(neighborhood_distribution_sliceA, torch.Tensor):
            ndA = neighborhood_distribution_sliceA.detach().cpu().numpy()
        else:
            ndA = np.asarray(neighborhood_distribution_sliceA)
        if isinstance(neighborhood_distribution_sliceB, torch.Tensor):
            ndB = neighborhood_distribution_sliceB.detach().cpu().numpy()
        else:
            ndB = np.asarray(neighborhood_distribution_sliceB)

        msd_neighborhood = pairwise_msd(ndA, ndB)
        M2 = nx.from_numpy(msd_neighborhood)

    else:
        raise ValueError(
            "Invalid neighborhood_dissimilarity. Expected one of {'jsd','cosine','msd'}; "
            f"got {neighborhood_dissimilarity!r}."
        )

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        if not isinstance(M1, torch.Tensor):
            M1 = nx.from_numpy(M1)
        if not isinstance(M2, torch.Tensor):
            M2 = nx.from_numpy(M2)
        M1 = M1.cuda()
        M2 = M2.cuda()
    
    # init distributions
    if a_distribution is None:
        # uniform distribution, a = array([1/n, 1/n, ...])
        a = nx.ones((sliceA.shape[0],))/sliceA.shape[0]
    else:
        a = nx.from_numpy(a_distribution)
        
    if b_distribution is None:
        b = nx.ones((sliceB.shape[0],))/sliceB.shape[0]
    else:
        b = nx.from_numpy(b_distribution)

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        a = a.cuda()
        b = b.cuda()
    
    if norm:
        # Heritage PASTE flag: scaled min distance to 1. 
        # Replaced globally by max-normalization [0,1] at distance calculation for stability.
        pass
    
    # Run OT
    if G_init is not None:
        G_init = nx.from_numpy(G_init)
        if isinstance(nx,ot.backend.TorchBackend):
            G_init = G_init.float()
            if use_gpu:
                G_init = G_init.cuda()

    G = nx.ones((a.shape[0], b.shape[0])) / (a.shape[0] * b.shape[0])
    if use_gpu and isinstance(nx, ot.backend.TorchBackend):
        G = G.cuda()

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    G_np = _to_np(G)

    if neighborhood_dissimilarity == 'jsd':
        initial_obj_neighbor = np.sum(_to_np(js_dist_neighborhood) * G_np)
    if neighborhood_dissimilarity == 'msd':
        initial_obj_neighbor = np.sum(_to_np(msd_neighborhood) * G_np)
    elif neighborhood_dissimilarity == 'cosine':
        initial_obj_neighbor = np.sum(_to_np(cosine_dist_neighborhood) * G_np)

    initial_obj_gene = np.sum(_to_np(cosine_dist_gene_expr) * G_np)

    if neighborhood_dissimilarity == 'jsd':
        # print(f"Initial objective neighbor (jsd): {initial_obj_neighbor}")
        logFile.write(f"Initial objective neighbor (jsd): {initial_obj_neighbor}\n")

    elif neighborhood_dissimilarity == 'cosine':
        # print(f"Initial objective neighbor (cosine_dist): {initial_obj_neighbor_cos}")
        logFile.write(f"Initial objective neighbor (cosine_dist): {initial_obj_neighbor}\n")
    elif neighborhood_dissimilarity == 'msd':
        # print(f"Initial objective neighbor (msd): {initial_obj_neighbor}")
        logFile.write(f"Initial objective neighbor (mean sq distance): {initial_obj_neighbor}\n")

    # print(f"Initial objective gene expr (cosine_dist): {initial_obj_gene}")
    logFile.write(f"Initial objective (cosine_dist): {initial_obj_gene}\n")
    

    # D_A: pairwise dist matrix of sliceA spots coords
    # a: initial distribution(uniform) of sliceA spots
    pi, logw = fused_gromov_wasserstein_incent(M1, M2, D_A, D_B, a, b, G_init = G_init, loss_fun='square_loss', alpha= alpha, gamma=gamma, log=True, numItermax=numItermax,verbose=verbose, use_gpu = use_gpu)
    pi = nx.to_numpy(pi)
    # obj = nx.to_numpy(logw['fgw_dist'])

    if neighborhood_dissimilarity == 'jsd':
        max_indices = np.argmax(pi, axis=1)
        # multiply each value of max_indices from pi_mat with the corresponding js_dist entry
        jsd_error = np.zeros(max_indices.shape)
        _dist_np = _to_np(js_dist_neighborhood)
        for i in range(len(max_indices)):
            jsd_error[i] = pi[i][max_indices[i]] * _dist_np[i][max_indices[i]]

        final_obj_neighbor = np.sum(jsd_error)
    elif neighborhood_dissimilarity == 'msd':
        final_obj_neighbor = np.sum(_to_np(msd_neighborhood)*pi)

    elif neighborhood_dissimilarity == 'cosine':
        max_indices = np.argmax(pi, axis=1)
        # multiply each value of max_indices from pi_mat with the corresponding js_dist entry
        cos_error = np.zeros(max_indices.shape)
        _dist_np = _to_np(cosine_dist_neighborhood)
        for i in range(len(max_indices)):
            cos_error[i] = pi[i][max_indices[i]] * _dist_np[i][max_indices[i]]

        final_obj_neighbor = np.sum(cos_error)


    final_obj_gene = np.sum(_to_np(cosine_dist_gene_expr) * pi)

    if neighborhood_dissimilarity == 'jsd':
        logFile.write(f"Final objective neighbor (jsd): {final_obj_neighbor}\n")
        # print(f"Final objective neighbor (jsd): {final_obj_neighbor}\n")
    elif neighborhood_dissimilarity == 'cosine':
        logFile.write(f"Final objective neighbor (cosine_dist): {final_obj_neighbor}\n")
        # print(f"Final objective neighbor (cosine_dist): {final_obj_neighbor}\n")

    logFile.write(f"Final objective gene expr(cosine_dist): {final_obj_gene}\n")
    # print(f"Final objective (cosine_dist): {final_obj_gene}\n")
    

    logFile.write(f"Runtime: {str(time.time() - start_time)} seconds\n")
    # print(f"Runtime: {str(time.time() - start_time)} seconds\n")
    logFile.write(f"---------------------------------------------\n\n\n")

    logFile.close()

    # new code ends

    if isinstance(backend,ot.backend.TorchBackend) and use_gpu:
        torch.cuda.empty_cache()

    if return_obj:
        return pi, initial_obj_neighbor, initial_obj_gene, final_obj_neighbor, final_obj_gene
    
    return pi


def neighborhood_distribution(curr_slice, radius):
    """
    This method is added by Anup Bhowmik
    Args:
        curr_slice: Slice to get niche distribution for.
        pairwise_distances: Pairwise distances between cells of a slice.
        radius: Radius of the niche.

    Returns:
        niche_distribution: Niche distribution for the slice.
    """

    cell_types = np.array(curr_slice.obs['cell_type_annot'].astype(str))
    unique_cell_types = np.unique(cell_types)
    cell_type_to_index = {ct: i for i, ct in enumerate(unique_cell_types)}
    
    source_coords = curr_slice.obsm['spatial']
    n_cells = curr_slice.shape[0]
    
    cells_within_radius = np.zeros((n_cells, len(unique_cell_types)), dtype=float)

    # Use BallTree instead of full O(n^2) distance matrix for memory & speed scalability
    from sklearn.neighbors import BallTree
    tree = BallTree(source_coords)
    neighbor_lists = tree.query_radius(source_coords, r=radius)

    for i in tqdm(range(n_cells), desc="Computing neighborhood distribution"):
        neighbors = neighbor_lists[i]
        for ind in neighbors:
            ct = cell_types[ind]
            cells_within_radius[i][cell_type_to_index[ct]] += 1
            
    # CRITICAL FIX: Normalize to probability distributions before computing JSD
    row_sums = cells_within_radius.sum(axis=1, keepdims=True)
    # Avoid division by zero for isolated cells
    row_sums[row_sums == 0] = 1 
    cells_within_radius = cells_within_radius / row_sums

    return cells_within_radius


def cosine_distance(sliceA, sliceB, sliceA_name, sliceB_name, filePath, use_rep = None, use_gpu = False, nx = ot.backend.NumpyBackend(), beta = 0.8, overwrite = False):
    from sklearn.metrics.pairwise import cosine_distances
    import os

    A_X, B_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceA,use_rep))), nx.from_numpy(to_dense_array(extract_data_matrix(sliceB,use_rep)))

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        A_X = A_X.cuda()
        B_X = B_X.cuda()

   
    s_A = A_X + 0.01
    s_B = B_X + 0.01

    fileName = f"{filePath}/cosine_dist_gene_expr_{sliceA_name}_{sliceB_name}.npy"
    
    if os.path.exists(fileName) and not overwrite:
        print("Loading precomputed Cosine distance of gene expression for slice A and slice B")
        cosine_dist_gene_expr = np.load(fileName)
        if use_gpu and isinstance(nx, ot.backend.TorchBackend):
            cosine_dist_gene_expr = torch.from_numpy(cosine_dist_gene_expr).cuda()
    else:
        print("Calculating cosine dist of gene expression for slice A and slice B")

        if isinstance(s_A, torch.Tensor) and isinstance(s_B, torch.Tensor):
            # Calculate manually using PyTorch to stay on GPU
            s_A_norm = s_A / s_A.norm(dim=1)[:, None]
            s_B_norm = s_B / s_B.norm(dim=1)[:, None]
            cosine_dist_gene_expr = 1 - torch.mm(s_A_norm, s_B_norm.T)
            np.save(fileName, cosine_dist_gene_expr.cpu().detach().numpy())
        else:
            from sklearn.metrics.pairwise import cosine_distances
            cosine_dist_gene_expr = cosine_distances(s_A, s_B)
            np.save(fileName, cosine_dist_gene_expr)

    return cosine_dist_gene_expr

