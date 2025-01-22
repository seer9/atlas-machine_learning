#!/usr/bin/env python3
"""
class that represents a poisson distribution
"""


class Poisson:
    """represents a Poisson distribution"""

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
            self.lambtha = float(sum(data) / len(data))

    """private method to calculate factorial of a number"""
    def _factorial(self, n):
        """
        calculates the factorial of 'k'
        """
        factorial = 1
        for i in range(1, n + 1):
            factorial *= i
        return factorial

    def pmf(self, k):
        """
        calculates the value of the PMF given 'k' for the number of successes
        """
        e = Poisson.e
        k = int(k)
        if k < 0:
            return 0
        return ((self.lambtha ** k)*(e ** -self.lambtha))/self._factorial(k)

    def cdf(self, k):
        """
        calculates the value of the CDF for a given number of successes
        """
        k = int(k)
        if k < 0:
            return 0
        F = 0
        """
        calculates for all values from 0 to the number of successes
        """
        for i in range(k + 1):
            F += self.pmf(i)
        return F
