#!/usr/bin/env python3
"""mean and covariance of a matrix"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a matrix.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the dataset.

    Returns:
        A tuple containing the mean and covariance of the dataset.
    """
    if type(X) is not np.ndarray:
        raise TypeError("X must be a numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    n, d = X.shape
    mean = np.mean(X, axis=0)
    mean = mean.reshape((1, d))
    cov = np.dot(X.T, X) / n

    return mean, cov
