#!/usr/bin/env python3
"""Maximization of a matrix"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization of a matrix.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the dataset.
        g: A numpy.ndarray of shape (k, n) containing the soft assignments.

    Returns:
        A numpy.ndarray of shape (k, d) representing the maximized matrix.
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None
    if type(g) is not np.ndarray or g.ndim != 2:
        return None, None, None
    if g.shape[1] != X.shape[0] or not np.allclose(g.sum(axis=0), 1.0):
        return None, None, None

    n, d = X.shape
    k, _ = g.shape
    
    pi = np.sum(g, axis=1) / n

    m = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]

    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = np.dot(g[i][:, np.newaxis] * diff.T, diff) / np.sum(g[i])

    return pi, m, S
