#!/usr/bin/env python3
"""K-means"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the dataset.
        k: The number of clusters.
        iterations: The maximum number of iterations to perform.

    Returns:
        A tuple containing the cluster centroids and the cluster indices.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    if k == 0:
        return None, None
    cent = np.random.uniform(low, high, size=(k, d))
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - cent, axis=2)
        closest = np.argmin(distances, axis=1)

        new_cent = np.array([X[closest == i].mean(axis=0)
                             for i in range(k)])
        if np.all(cent == new_cent):
            break
        cent = new_cent
    
    return cent, closest
