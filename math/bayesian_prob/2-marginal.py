#!/usr/bin/env python3
"""likelihood matrix"""
import numpy as np


def intersection(x, n, P, Pr):
    """
    calculates the marginal probability of obtaining the data.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the data points.
        n: The number of data points.
        P: A numpy.ndarray of shape (d, d) representing the covariance matrix.
        Pr: A numpy.ndarray of shape (d, d) containing the prior beliefs of P

    Returns:
        the marginal probability of obtaining x and n
    """
    factorial = np.math.factorial
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        str = "x must be an integer that is greater than or equal to 0"
        raise ValueError(str)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    patience = np.math.factorial(n)
    ill = np.math.factorial(x) * np.math.factorial(n - x)
    coef = patience / ill
    likelihood = coef * (P ** x) * ((1 - P) ** (n - x))
    intersection = likelihood * Pr
    return np.sum(intersection)
