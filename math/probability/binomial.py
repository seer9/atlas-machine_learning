#!/usr/bin/env python3
"""represent a binomial distribution"""


class Binomial:
    """
        binomial distribution is used for calculating the probability of
        getting a number of successes in a fixed number of trials
    """

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

    def _factorial(self, n):
        """
        calculates the factorial of 'n'
        """
        product = 1
        for i in range(1, n + 1):
            product *= i
        return product
    
    def pmf(self, k):
        """
        calculates the value of the PMF for a given number of successes
        """
        if type(k) is not int:
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        fact_n = self._factorial(self.n)
        fact_k = self._factorial(k)
        fact_nk = self._factorial(self.n - k)
        nkn = fact_n / (fact_k * fact_nk)
        return nkn * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
        calculates the value of the CDF for a given number of successes
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
