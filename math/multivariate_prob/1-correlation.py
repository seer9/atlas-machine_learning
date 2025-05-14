#!/usr/bin/env python3
"""Correlation of a matrix"""
import numpy as np


def correlation(C):
    """
    Calculates the correlation of a covariance matrix.

    Args:
        C: A numpy.ndarray of shape (d, d) representing the covariance matrix.

    Returns:
        A numpy.ndarray of shape (d, d) representing the correlation matrix.
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a non-empty square matrix")

    v = np.diag(C)
    sd = np.sqrt(v)

    return C / np.outer(sd, sd)
