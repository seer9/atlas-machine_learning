#!/usr/bin/env python3
import scipy.cluster.hierarchy


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on the dataset X.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the data points.
        dist: The distance metric to use for clustering.

    Returns:
        Z: A linkage matrix representing the hierarchical clustering.
    """
    clss = scipy.cluster.hierarchy.linkage(X, method='ward', metric=dist)
    return clss
