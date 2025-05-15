#!/usr/bin/env python3
"""Initialize K-means"""
import numpy as np


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
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k < 0:
        return None

    n, d = X.shape
    if k > n or type(k) is not int or k <= 0:
        return None
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    centroids = np.random.uniform(low, high, (k, d))
    return centroids
