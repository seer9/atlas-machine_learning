#!/usr/bin/env python3
"""
function to manipulate 2D matrices
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    concatenates two matrices along a specific axis
    """
    if not all(len(row) == len(mat1[0]) for row in mat1):
        return None
    if not mat1 or not mat2:
        return []
    if not mat1:
        return mat2 if axis == 0 else None
    if not mat2:
        return mat1 if axis == 0 else None

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        else:
            return mat1 + mat2

    elif axis == 1:
        if len(mat1) == len(mat2):
            return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
        else:
            return None
