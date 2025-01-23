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
