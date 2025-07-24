# utils/io.py
import os

import numpy as np
import pandas as pd
import torch, logging
import pickle
from scipy.sparse import linalg
import scipy.sparse as sp
from sklearn.metrics.pairwise import haversine_distances


def move_to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [move_to_device(b, device) for b in batch]
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    return batch.to(device, non_blocking=True)

epsilon = 1e-8
logger  = logging.getLogger(__name__)


def calculate_symmetric_normalized_laplacian(adj):
    r"""
    Description:
    -----------
    Calculate Symmetric Normalized Laplacian.
    Assuming unnormalized laplacian matrix is `L = D - A`,
    then symmetric normalized laplacian matrix is:
    `L^{Sym} =  D^-1/2 L D^-1/2 =  D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2`
    For node `i` and `j` where `i!=j`, L^{sym}_{ij} <=0.

    Parameters:
    -----------
    adj: np.ndarray
        Adjacent matrix A

    Returns:
    -----------
    symmetric_normalized_laplacian: np.matrix
        Symmetric normalized laplacian L^{Sym}
    """
    adj                                 = sp.coo_matrix(adj)
    D                                   = np.array(adj.sum(1))
    D_inv_sqrt = np.power(D, -0.5).flatten()    # diagonals of D^{-1/2}
    D_inv_sqrt[np.isinf(D_inv_sqrt)]    = 0.
    matrix_D_inv_sqrt                   = sp.diags(D_inv_sqrt)   # D^{-1/2}
    symmetric_normalized_laplacian      = sp.eye(adj.shape[0]) - matrix_D_inv_sqrt.dot(adj).dot(matrix_D_inv_sqrt).tocoo()
    return symmetric_normalized_laplacian

def calculate_scaled_laplacian(adj, lambda_max=2, undirected=True):
    r"""
    Description:
    -----------
    Re-scaled the eigenvalue to [-1, 1] by scaled the normalized laplacian matrix for chebyshev pol.
    According to `2017 ICLR GCN`, the lambda max is set to 2, and the graph is set to undirected.
    Note that rescale the laplacian matrix is equal to rescale the eigenvalue matrix.
    `L_{scaled} = (2 / lambda_max * L) - I`

    Parameters:
    -----------
    adj: np.ndarray
        Adjacent matrix A

    Returns:
    -----------
    L_res: np.matrix
        The rescaled laplacian matrix.
    """
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    L       = calculate_symmetric_normalized_laplacian(adj)
    if lambda_max is None:  # manually cal the max lambda
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L       = sp.csr_matrix(L)
    M, _    = L.shape
    I       = sp.identity(M, format='csr', dtype=L.dtype)
    L_res   = (2 / lambda_max * L) - I
    return L_res

def symmetric_message_passing_adj(adj):
    r"""
    Description:
    -----------
    Calculate the renormalized message passing adj in `GCN`.

    Parameters:
    -----------
    adj: np.ndarray
        Adjacent matrix A

    Returns:
    -----------
    mp_adj:np.matrix
        Renormalized message passing adj in `GCN`.
    """
    # add self loop
    print("calculating the renormalized message passing adj, please ensure that self-loop has added to adj.")
    adj         = sp.coo_matrix(adj)
    rowsum      = np.array(adj.sum(1))
    d_inv_sqrt  = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt  = sp.diags(d_inv_sqrt)
    mp_adj          = d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()
    return mp_adj

def transition_matrix(adj):
    r"""
    Description:
    -----------
    Calculate the transition matrix `P` proposed in DCRNN and Graph WaveNet.
    P = D^{-1}A = A/rowsum(A)

    Parameters:
    -----------
    adj: np.ndarray
        Adjacent matrix A

    Returns:
    -----------
    P:np.matrix
        Renormalized message passing adj in `GCN`.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    # P = d_mat.dot(adj)
    P = d_mat.dot(adj).astype(np.float32).todense()
    return P



def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_distance_matrix(distance_path):
    path = os.path.join(distance_path, 'dist.npy')
    try:
        dist = np.load(path)
    except:
        distances = pd.read_csv(os.path.join(distance_path, 'distances.csv'))
        with open(os.path.join(distance_path, 'graph_sensor_ids.txt')) as f:
            ids = f.read().strip().split(',')
        num_sensors = len(ids)
        dist = np.ones((num_sensors, num_sensors), dtype=np.float32) * np.inf
        # Builds sensor id to index map.
        sensor_id_to_ind = {int(sensor_id): i for i, sensor_id in enumerate(ids)}

        # Fills cells in the matrix with distances.
        for row in distances.values:
            if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                continue
            dist[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]
        np.save(path, dist)
    return dist

def get_similarity(thr=0.1,distance_path=None,force_symmetric=False, sparse=False):
    dist = load_distance_matrix(distance_path)
    finite_dist = dist.reshape(-1)
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < thr] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj

def geographical_distance(x=None, to_rad=True):
    """
    Compute the as-the-crow-flies distance between every pair of samples in `x`. The first dimension of each point is
    assumed to be the latitude, the second is the longitude. The inputs is assumed to be in degrees. If it is not the
    case, `to_rad` must be set to False. The dimension of the data must be 2.

    Parameters
    ----------
    x : pd.DataFrame or np.ndarray
        array_like structure of shape (n_samples_2, 2).
    to_rad : bool
        whether to convert inputs to radians (provided that they are in degrees).

    Returns
    -------
    distances :
        The distance between the points in kilometers.
    """
    _AVG_EARTH_RADIUS_KM = 6371.0088

    # Extract values of X if it is a DataFrame, else assume it is 2-dim array of lat-lon pairs
    latlon_pairs = x.values if isinstance(x, pd.DataFrame) else x

    # If the input values are in degrees, convert them in radians
    if to_rad:
        latlon_pairs = np.vectorize(np.radians)(latlon_pairs)

    distances = haversine_distances(latlon_pairs) * _AVG_EARTH_RADIUS_KM

    # Cast response
    if isinstance(x, pd.DataFrame):
        res = pd.DataFrame(distances, x.index, x.index)
    else:
        res = distances

    return res

def thresholded_gaussian_kernel(x, theta=None, threshold=None, threshold_on_input=False):
    if theta is None:
        theta = np.std(x)
    weights = np.exp(-np.square(x / theta))
    if threshold is not None:
        mask = x > threshold if threshold_on_input else weights < threshold
        weights[mask] = 0.
    return weights

def get_air_similarity(thr=0.1, include_self=False, force_symmetric=False, sparse=False, path = None,**kwargs):
    stations = pd.DataFrame(pd.read_hdf(path, 'stations'))
    st_coord = stations.loc[:, ['latitude', 'longitude']]
    dist = geographical_distance(st_coord, to_rad=True).values
    theta = np.std(dist[:36, :36])  # use same theta for both air and air36
    adj = thresholded_gaussian_kernel(dist, theta=theta, threshold=thr)
    if not include_self:
        adj[np.diag_indices_from(adj)] = 0.
    if force_symmetric:
        adj = np.maximum.reduce([adj, adj.T])
    if sparse:
        import scipy.sparse as sps
        adj = sps.coo_matrix(adj)
    return adj

def load_adj(pickle_file,adj_type):
    r"""
    Description:
    -----------
    Load pickle data.

    Parameters:
    -----------
    pickle_file: str
        File path.

    Returns:
    -----------
    pickle_data: any
        Pickle data.
    """
    try:
        # METR and PEMS_BAY
        sensor_ids, sensor_id_to_ind, ori_adj = load_pickle(pickle_file)
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    if adj_type == "scalap":
        adjs = [calculate_scaled_laplacian(ori_adj).astype(np.float32).todense()]
    elif adj_type == "normlap":
        adjs = [calculate_symmetric_normalized_laplacian(ori_adj).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        adjs = [symmetric_message_passing_adj(ori_adj).astype(np.float32).todense()]
    elif adj_type == "transition":
        adjs = [transition_matrix(ori_adj).T]
    elif adj_type == "doubletransition":
        adjs = [transition_matrix(ori_adj).T, transition_matrix(ori_adj.T).T]
    elif adj_type == "identity":
        adjs = [np.diag(np.ones(ori_adj.shape[0])).astype(np.float32).todense()]
    elif adj_type == 'original':
        adjs = ori_adj
    else:
        error = 0
        assert error, "adj type not defined"
    return adjs, ori_adj
