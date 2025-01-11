#!/usr/bin/env python3
"""
funtion to multipy matrices
"""


def mat_mul(mat1, mat2):
    """
    multiplies two matrices if they are the same size
    """
    if len(mat1[0]) != len(mat2):
        return None
    return [[sum(a * b for a, b in zip(row_a, col_b))
             for col_b in zip(*mat2)]
            for row_a in mat1]
