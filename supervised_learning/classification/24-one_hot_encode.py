#!/usr/bin/env python3
"""module to perform one hot encoding"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    converts a numeric label vector into a one-hot matrix
    """
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int) or classes <= 0:
        return None

    one_hot = np.zeros((classes, Y.shape[0]))
    one_hot[Y, np.arange(Y.shape[0])] = 1
    return one_hot
