#!/usr/bin/env python3
"""
represents a normal distribution
"""


class Normal:
    """
    represents a normal distribution.
    data - list of the data to be used to estimate the distribution
    mean - mean of the distribution
    stddev - standard deviation of the distribution
    """

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            """
            variance = sum((x - mean) ** 2) / n
            """
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """
        calculates the z-score of a given x-value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        calculates the x-value of a given z-score
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        calculates the value of the PDF for a given x-value
        """

        e = Normal.e
        pi = Normal.pi
        coef = 1 / (self.stddev * (2 * pi) ** 0.5)
        expo = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)
        return coef * (e ** expo)

    def _erf(self, z):
        """
        calculates the error function
        """
        pi = Normal.pi
        erf = z - (z ** 3) / 3 + (z ** 5) / 10 - (z ** 7) / 42 + (z ** 9) / 216
        return 2 * erf / (pi ** 0.5)

    def cdf(self, x):
        """
        calculates the value of the CDF for a given x-value
        """

        e = Normal.e
        x = (x - self.mean) / (self.stddev * (2 ** 0.5))
        return (1 + self._erf(x)) / 2
