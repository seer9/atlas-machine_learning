#!/usr/bin/env python3
"""
module for the function poly_derivative(poly)
"""


def poly_derivative(poly):
    """
    takes in a representing a polynomial
    and returns the derivative of the polynomial
    """
    if not isinstance(poly, list) or len(poly) == 0:
        """if poly is not a list or is an empty list"""
        return None
    """return the derivative of the polynomial"""
    return [poly[i] * i for i in range(1, len(poly))]
