#!/usr/bin/env python3
"""
module for the function poly_integral(poly)
"""


def poly_integral(poly, C=0):
    """
    takes in a  poly and returns the integral of poly
    """
    if not isinstance(poly, list) or not  isinstance(C, int):
        """if poly is not a list or is empty"""
        return None
    if not isinstance(C, (int, float)):
        """if C is not a float"""
        return None
    if len(poly) == 1:
        """if poly has only one element"""
        return [C]
    """return the integral of the polynomial"""
    return [C] + [poly[i] / (i + 1) for i in range(len(poly))]

