#!/usr/bin/env python3
"""function to perform PCA"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the dataset.
        ndim: The new dimensionality of the transformed data.

    Returns:
        A numpy.ndarray of shape (d, ndim) containing the projection matrix,
        or None if the input is invalid.
    """
    centered = X - np.mean(X, axis=0)
    lv, sv, rv = np.linalg.svd(centered)
    w = rv[:ndim].T
    return np.matmul(centered, w)
