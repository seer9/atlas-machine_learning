#!/usr/bin/env python3
"""K-means clustering algorithm"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means clustering on the dataset X.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the data points.
        k: The number of clusters.

    Returns:
        centroids: A numpy.ndarray of shape (k, d) containing the cluster
                   centroids.
        labels: A numpy.ndarray of shape (n,) containing the index of the
                cluster each data point belongs to.
    """
    if not isinstance(X, (list, tuple)):
        return None, None
    if not all(isinstance(x, (list, tuple)) for x in X):
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None

    model = sklearn.cluster.KMeans(n_clusters=k)
    model.fit(X)

    C = model.cluster_centers_
    clss = model.labels_

    return C, clss
