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


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the dataset.
        k: A positive integer containing the number of clusters.
        iterations: maximum number of iterations.

    Returns:
        A tuple containing:
            - C: A numpy.ndarray containing the centroid means for all cluster.
            - clss: A numpy.ndarray containing the index of the cluster in C
              that each data point belongs to.
        Or (None, None) on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    if k > n:
        return None, None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    centroids = np.random.uniform(low, high, (k, d))

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clss = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[clss == i].mean(axis=0)
            if np.any(clss == i)
            else np.random.uniform(low, high)
            for i in range(k)
        ])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids, clss
