#!/usr/bin/env python3
"""Variance of a matrix"""
import numpy as np


def variance(X, C):
    """
    Calculates the variance of a matrix.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the dataset.
        C: A numpy.ndarray of shape (d, d) representing the covariance matrix.

    Returns:
        A numpy.ndarray of shape (d,) containing the variance of each feature.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return
    if X.shape[1] != C.shape[1]:
        return None

    distance = np.min(np.linalg.norm(X[:, np.newaxis], axis=-1), axis=-1)

    return (distance ** 2) / X.shape[0]
