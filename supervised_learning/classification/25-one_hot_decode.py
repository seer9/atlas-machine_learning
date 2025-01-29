#!/usr/bin/env python3
"""module to perform one hot decoding"""
import numpy as np


def one_hot_decode(one_hot):
    """
    converts a one-hot matrix into a vector of labels
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    return np.argmax(one_hot, axis=0)
