#!/usr/bin/env python3
"""Gaussian Mixture Model"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian Mixture Model.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the data points.
        m: A numpy.ndarray of shape (k, d) containing the cluster means.
        S: A numpy.ndarray of shape (k, d, d) containing the covariance matrices.

    Returns:
        A numpy.ndarray of shape (k, n) containing the probability density
        function for each cluster and data point.
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None
    if type(m) is not np.ndarray or m.ndim != 1:
        return None
    if type(S) is not np.ndarray or S.ndim != 2:
        return None
    if X.shape[1] != m.shape[0] or S.shape[0] != S.shape[1]:
        return None
    if S.shape[0] != m.shape[0]:
        return None

    n, d = X.shape
    
    det = np.linalg.det(S)
    if det == 0:
        return None
    
    inverse = np.linalg.inv(S)

    norm = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det))

    diff = X[:, np.newaxis] - m
    exponent = -0.5 * np.sum(diff @ inverse * diff, axis=1)

    pdf = norm * np.exp(exponent)

    return pdf
