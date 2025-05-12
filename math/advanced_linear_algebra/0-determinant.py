#!/usr/bin/env python3


def determinant(matrix):
    """
    Calculates the determinant of a matrix.

    Args:
        matrix: A square matrix.

    Returns:
        The determinant of the matrix.
    """
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be square")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for c in range(len(matrix)):
        det += ((-1) ** c) * matrix[0][c] * determinant(
            [row[:c] + row[c + 1:] for row in matrix[1:]])

    return det
