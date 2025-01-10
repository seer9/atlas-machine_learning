#!/usr/bin/env python3
"""
function that transposes a matrix
"""


def matrix_transpose(matrix):
    """
    function that transposes a matrix
    """

    return [[matrix[j][i] for j in range(len(matrix))]
            for i in range(len(matrix[0]))]
