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
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    cent = np.random.uniform(low, high, (k, d))

    for _ in range(iterations):
        closest = np.argmin(np.linalg.norm(X[:, None] - cent, axis=-1), axis=1)
        new = np.copy(cent)
        for i in range(k):
            if i not in closest:
                new[i] = np.random.uniform(low, high, (d,))
            else:
                new[i] = np.mean(X[closest == i], axis=0)
        if np.array_equal(new, cent):
            break
        cent = new
    return cent, closest
