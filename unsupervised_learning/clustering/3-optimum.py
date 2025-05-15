#!/usr/bin/env python3
"""optimal K-means"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum(X, kmin=1, kmax=None, iterations=1000):
    """
    Calculates the optimal number of clusters in a dataset using K-means.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the dataset.
        kmin: The minimum number of clusters to check.
        kmax: The maximum number of clusters to check.
        iterations: The maximum number of iterations for K-means.

    Returns:
        A tuple containing:
            - results: A list containing the variance of each cluster for each
              number of clusters.
            - d_vars: A list containing the difference in variance between each
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax < kmin):
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if type(kmax) is not int and kmax <= kmin:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    results = []
    d_vars = []

    C, clss = kmeans(X, kmin, iterations)
    base = variance(X, C)
    results.append((C, clss))

    d_vars = [0.0]
    k = kmin + 1

    while k <= kmax:
        C, clss = kmeans(X, k, iterations)
        var = variance(X, C)
        results.append((C, clss))
        d_vars.append(base - var)
        base = var
        k += 1

    return results, d_vars
