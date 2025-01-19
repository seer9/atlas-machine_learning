#!/usr/bin/env python3
"""
module for calculating the sum of the squares of natural numbers
"""


def summation_i_squared(n):

    """
    takes n as argument and returns the sum of
    the square roots of natural numbers
    """
    if not isinstance(n, int) or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
