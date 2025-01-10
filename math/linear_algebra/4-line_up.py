#!/usr/bin/env python3
"""
function that adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
    function that returns the sum of two arrays or
    none if they are not the same size
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
