#!/usr/bin/env python3
"""
function to manipulate 2D matrices
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    concatenates two matrices along a specific axis
    """
    if axis == 0:
        return [row.copy() for row in mat1] + [row.copy() for row in mat2]
    if axis == 1:
        return [row.copy() + mat2[i].copy() for i, row in enumerate(mat1)]
    return None
