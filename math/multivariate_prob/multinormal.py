#!/usr/bin/env python3
"""Multivariate normal distribution"""
import numpy as np


class MultiNormal:
    """represents a multivariate normal distribution"""
    def __init__(self, data):
        """initializes the distribution"""
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1).reshape(-1, 1)
        mean = self.mean
        self.cov = np.matmul(
            data - mean, (data - mean).T) / (data.shape[1] - 1)

    def pdf(self, x):
        """calculates the PDF at a given point"""
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        const = 1 / ((2 * np.pi) ** (d / 2) * np.linalg.det(self.cov) ** 0.5)
        dev = x - self.mean
        inv = np.linalg.inv(self.cov)
        exponent = -0.5 * np.matmul(np.matmul(dev.T, inv), dev)
        return const * np.exp(exponent[0, 0])
