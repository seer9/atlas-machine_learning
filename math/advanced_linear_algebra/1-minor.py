#!/usr/bin/env python3
"""minor of a matrix"""


def minor(matrix):
    """
    Calculates the minor of a matrix.

    Args:
        matrix: A square matrix.

    Returns:
        The minor of the matrix.
    """
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        if matrix == [[]]:
            return 1
        raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return [[1]]
    if len(matrix) == 2:
        return [[matrix[1][1]], [matrix[0][1]]]
    minors = []
    for i in range(len(matrix)):
        minors.append([row[:i] + row[i + 1:] for row in matrix[1:]])

    return minors