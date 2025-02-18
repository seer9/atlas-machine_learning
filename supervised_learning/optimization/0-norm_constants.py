#!/usr/bin/env python3
"""nornaization constants"""
import numpy as np


def normalization_constants(X):
    """
    calculates the normalization constants of a matrix
    perams: X: is the numpy.ndarray of shape (m, nx) to normalize
    return: mean and standard deviation of each feature
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
