#!/usr/bin/env python3
"""
function that calculates the shape of a matrix
"""


def matrix_shape(matrix):

    """
    takes in a matrix and returns the shape as a list of integers
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
