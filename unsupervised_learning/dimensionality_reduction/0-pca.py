#!/usr/bin/env python3
"""function for PCA on a dataset"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the dataset.
        var: variance that the PCA transformation should maintain.

    Returns:
        A numpy.ndarray of shape (d, d') containing the projection matrix,
        where d' is the new dimensionality of the transformed data.
    """
    U, S, V = np.linalg.svd(X)
    acum = np.cumsum(S)
    dim = []
    for i in range(len(S)):
        if acum[i] / np.sum(S) >= var:
            dim.append(i)
    r = dim[0] + 1
    return V[:r].T
