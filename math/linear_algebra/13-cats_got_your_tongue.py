#!/usr/bin/env python3
import numpy as np
"""
function that concatenates two matrices along a specific axis
"""


def np_cat(mat1, mat2, axis=0):
    """
    returns the concatenation of two matrices
    """
    return np.concatenate((mat1, mat2), axis)