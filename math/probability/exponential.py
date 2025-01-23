#!/usr/bin/env python3
"""
class to represent an exponential distribution
"""


class Exponential:
    """
    Class Exponential that represents an exponential distribution
    """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        if data is not given, lambtha is used
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            """
            data is given, calculate the lambtha of data;
            if data is not a list and/or doesnt have 2 values,
            raise respective errors.
            """
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """
        calculates the value of the PDF for a given time period (x).
        """

        """
        if time period is less than 0, return 0
        """
        e = Exponential.e
        if x < 0:
            return 0

        """
        formula for pdf of exponential distribution
        """
        return self.lambtha * (e ** (-self.lambtha * x))

    def cdf(self, x):
        """
        calculates the value of the CDF for a given time period (x).
        """

        """
        if time period is less than 0, return 0
        """
        e = Exponential.e
        if x < 0:
            return 0

        """
        formula for cdf of exponential distribution
        """
        return 1 - e ** (-self.lambtha * x)
