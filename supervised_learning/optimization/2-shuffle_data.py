#!/usr/bin/env python3
"""shuffles the data points in two matrices the same way"""
import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices the same way
    returns: the shuffled X and Y matrices
    """
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    return X[shuffle], Y[shuffle]
