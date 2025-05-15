#!/usr/bin/env python3
"""Initialize K-means"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes K-means with k random cluster centroids.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the dataset.
        k: The number of clusters.

    Returns:
        A numpy.ndarray of shape (k, d) containing the initialized centroids,
        or None if the input is invalid.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    pi = np.full((k,), 1 / k)

    m, _ = kmeans(X, k)

    if m is None:
        return None, None, None

    S = np.tile(np.eye(X.shape[1]), (k, 1, 1))

    return pi, m, S
