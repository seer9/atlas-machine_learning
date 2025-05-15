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
        raise TypeError("X must be a 2D numpy.ndarray")
    if type(C) is not np.ndarray or len(C.shape) != 2:
        raise TypeError("C must be a 2D numpy.ndarray")
    if X.shape[1] != C.shape[0]:
        raise ValueError("X and C must have compatible shapes")

    return np.diagonal(C)
