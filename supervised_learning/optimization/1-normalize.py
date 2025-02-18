#!/usr/bin/env python3
"""normalizes a matrix"""


def normalize(X, m, s):
    """
    normalizes a matrix
    param X: to normalize
    param m: contains the mean of all features of X
    param s: contains the standardization of all features of X
    returns:  the normalized X matrix
    """
    return (X - m) / s
