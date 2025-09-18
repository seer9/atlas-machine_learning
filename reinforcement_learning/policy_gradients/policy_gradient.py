#!/usr/bin/env python3
"""Policy Gradient"""
import numpy as np


def policy(matrix, weight):
    """computes the policy with a weight of a matrix.
    Args:
        matrix: the number of examples, n is the number of features.
        weight: where k is the number of classes.
    Returns: the softmax probabilities for each class, respectively.
    """
    z = matrix @ weight
    # compute softmax
    exp = np.exp(z - np.max(z))
    return exp / np.sum(exp)
