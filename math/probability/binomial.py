#!/usr/bin/env python3
"""represent a binomial distribution"""


class Binomial:

    def __init__(self, data=None, n=1, p=0.5):
        """
        initializing method.
        data - estimates the distribution
        n - number of Bernoulli trials
        p - probability of success
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = n
            self.p = p
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum([(x - mean) ** 2 for x in data]) / len(data)
            p = 1 - variance / mean
            n = round(mean / p)
            p = mean / n
            self.n = n
            self.p = p
