#!/usr/bin/env python3
"""
module for the function poly_integral(poly)
"""


def poly_integral(poly, C=0):
    """
    if poly is not a list or is an empty list, return None
    """
    if type(poly) is not list or len(poly) == 0:
        return None
    elif type(C) is int:
        """
        if poly is a list and C is an integer
        """
        if poly == [0]:
            return [C]
        power = 0
        integral = poly.copy()
        """i is the index of the element in the list"""
        for i in range(len(integral)):
            if type(integral[i]) is int or type(integral[i]) is float:
                power += 1
                num = integral[i] / power
                integral[i] = int(num) if num % 1 == 0 else num
            else:
                return None
        integral.insert(0, C)
        """return the integral of the polynomial"""
        return integral
    else:
        return None
