#!/usr/bin/env python3
"""definiteness of a matrix"""
import numpy as np


def definiteness(matrix):
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if matrix.shape[0] != matrix.shape[1]:
        return None
    trans = np.transpose(matrix)
    if not np.array_equal(trans, matrix):
        return None
    ev, _ = np.linalg.eig(matrix)
    if all(ev < 0):
        return "Negative definite"
    elif all(ev <= 0):
        return "Negative semi-definite"
    elif all(ev > 0):
        return "Positive definite"
    elif all(ev >= 0):
        return "Positive semi-definite"
    else:
        return "Indefinite"