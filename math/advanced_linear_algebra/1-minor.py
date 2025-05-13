#!/usr/bin/env python3
"""derterminant of a matrix"""


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
        if matrix == [[]]:
            return 1
        raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for c in range(len(matrix)):
        det += ((-1) ** c) * matrix[0][c] * determinant(
            [row[:c] + row[c + 1:] for row in matrix[1:]])

    return det


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
    for i in matrix:
        if len(matrix) != len(i):
            raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]
    minors = []
    for i in range(len(matrix)):
        minors.append([])
        for j in range(len(matrix)):
            minors[i].append(determinant(
                [row[:j] + row[j + 1:]
                 for row in matrix[:i] + matrix[i + 1:]]))
    return minors
