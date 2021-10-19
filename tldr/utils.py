# Copyright (c) NAVER and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from time import time

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def get_knn_graph(
    X, n_neighbors, l2_norm_graph=False, device="cuda", metric="IP", verbose=0, knn_approximation=None
):
    """Computes and returns the k nearest neighbours of each sample
    Parameters
    ----------
    X : ndarray
        N x D array containing N samples of dimension D
    n_neighbors : int
        Number of nearest neighbors
    l2_norm_graph : bool
        L2 normalize samples
    device : str
        Selects the device [cpu, gpu]
    metric : str
        Selects the similarity metric [IP, L2]
    verbose : int
        Selects verbosity level [0, 1, 2]
    knn_approximation : str
        Enables nearest neighbor approximation [None, low, medium, high]
    Returns
    -------
    knn_graph : ndarray
        Array containing the indices of the k nearest neighbors of each sample
    """
    knn_graph = None
    metric = metric.upper()
    if metric not in ["IP", "L2"]:
        raise ValueError(f"similarity metric {metric} not supported. Metrics supported are 'L2' and 'IP'")

    if verbose > 1:
        print(f" - Creating {n_neighbors}-NN graph for training data")
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    X = X.astype(np.float32)
    X = np.ascontiguousarray(X)
    if l2_norm_graph:
        X = torch.nn.functional.normalize(torch.tensor(X), dim=1, p=2).cpu().numpy()

    split_train = np.array_split(X, 100)
    all_neighbors = list()
    all_dists = list()
    dimensions = X.shape[1]
    used_dimensions = dimensions
    faiss_type = "Flat"
    if metric == "IP":
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    elif metric == "L2":
        faiss_metric = faiss.METRIC_L2
    if knn_approximation is not None:
        if knn_approximation not in ["low", "medium", "high"]:
            raise ValueError(
                f"knn_approximation should be one of None, low, medium or high and it was {knn_approximation}"
            )
        approx_mapping = {"low": 32, "medium": 16, "high": 8}
        used_dimensions = approx_mapping[knn_approximation]
        if (dimensions % used_dimensions) != 0:
            raise ValueError(
                f"Number of dimensions of training data must be divisible by {used_dimensions} to allow {knn_approximation} knn_approximation"
            )
        faiss_type = f"IVF1,PQ{used_dimensions}"
    faiss_index = faiss.index_factory(dimensions, faiss_type, faiss_metric)
    if "cuda" in device and faiss.get_num_gpus() > 0:
        faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)

    t0 = time()
    if knn_approximation is not None:
        if verbose > 1:
            print(
                "Training product quantization for knn approximation with 10% of total data. Note that for small datasets training + computing neighbors could be slower than brute force computation"
            )
        faiss_index.train(X[: 1 + X.shape[0] // 10])
        if verbose > 1:
            print("Finished training product quantization")
    faiss_index.add(X)  # add vectors to the index
    with get_progress_bar() as progress:
        if verbose > 0:
            task = progress.add_task(description="[green]Computing KNN", total=len(split_train), info="-")
        for splitted in split_train:
            D, Idx = faiss_index.search(
                splitted, k=n_neighbors + 1
            )  # n_neighbors+1 because the first one is always yourself...
            all_neighbors.append(Idx[:, 1:])
            all_dists.append(D[:, 1:])
            if verbose > 0:
                progress.update(task, advance=1)
    t1 = time()
    if verbose > 1:
        print(" - KNN computation took %.2g sec" % (t1 - t0))
    knn_graph = np.concatenate(all_neighbors, axis=0)

    knn_graph = knn_graph[:, :n_neighbors]

    return knn_graph


def tonumpy(x):
    """Converts a tensor to numpy array"""
    if type(x).__module__ == torch.__name__:
        return x.cpu().numpy()
    else:
        return x


def whiten(X, fudge=1e-18):
    """Applies whitening to an NxD array of N samples with dimensionality D"""
    # the matrix X should be observations-by-components

    # get the covariance matrix
    Xcov = np.dot(X.T, X)
    # eigenvalue decomposition of the covariance matrix
    d, V = np.linalg.eigh(Xcov)

    # a fudge factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified.
    D = np.diag(1.0 / np.sqrt(d + fudge))

    # whitening matrix
    W = np.dot(np.dot(V, D), V.T)

    # multiply by the whitening matrix
    X_white = np.dot(X, W)

    return X_white, W


def l2_normalize(x, axis=-1):
    """L2 normalizes an NxD array of N samples with dimensionality D"""
    x = F.normalize(x, p=2, dim=axis)
    return x


def parse_net_config(net_config: str):
    """Parses an architecture configuration string and returns the corresponding network type, number of hidden layers and their dimensionality"""
    config = net_config.split("-")

    net_type = config[0].lower()
    if net_type not in ["linear", "flinear", "mlp"]:
        raise ValueError(
            f"Incorrect network configuration format '{net_config}': incorrect network type '{net_type}', currently supported types are 'linear', 'flinear', and 'mlp'"
        )

    if len(config) == 1:
        if net_type not in ["linear"]:
            raise ValueError(
                f"Incorrect network configuration format '{net_config}': you need to specify the number of layers and dimensionality of each layer `{net_type}-[NUM_HLAYERS]-[HDIMS]`"
            )
        return net_type, 0, []

    num_hidden_layers = int(config[1])
    if len(config) == 2:
        if num_hidden_layers == 0:
            return net_type, num_hidden_layers, []
        raise ValueError(
            f"Incorrect network configuration format '{net_config}': you need to specify the dimensionality of each hidden layer using `{net_type}-{num_hidden_layers}-[HDIMS]`"
        )

    hidden_layers_dim = [int(e) for e in config[2:]]
    return net_type, num_hidden_layers, hidden_layers_dim


def get_progress_bar():
    """Returns a progress bar using the rich library"""
    return Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[bold blue]{task.fields[info]}", justify="right"),
        TimeRemainingColumn(),
    )
