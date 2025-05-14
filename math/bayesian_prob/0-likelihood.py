#!/usr/bin/env python3
"""likelihood matrix"""
import numpy as np


def likelihood(X, n, P):
    """
    Calculates the likelihood of a data point given a set of parameters.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the data points.
        n: The number of data points.
        P: A numpy.ndarray of shape (d, d) representing the covariance matrix.

    Returns:
        A numpy.ndarray of shape (n, 1) containing the likelihood of each data point.
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(X) is not int or X < 0:
        str = "x must be an integer that is greater than or equal to 0"
        raise TypeError(str)
    if X > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    
    patience = np.math.factorial(n)
    ill = np.math.factorial(X) * np.math.factorial(n - X)
    coef = patience / ill
    return coef * (P ** X) * ((1 - P) ** (n - X))
