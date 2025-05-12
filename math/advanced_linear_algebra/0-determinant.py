#!/usr/bin/env python3
import numpy as np


def determinant(matrix):
    """
    Calculates the determinant of a matrix.

    Args:
        matrix: A square matrix.

    Returns:
        The determinant of the matrix.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2:
        raise ValueError("matrix must be a 2D array")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a square matrix")

    return np.linalg.det(matrix)
