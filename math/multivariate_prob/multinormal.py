#!/usr/bin/env python3
"""Multivariate normal distribution"""
import numpy as np


class MultiNormal:
    """represents a multivariate normal distribution"""
    def __init__(self, data):
        """initializes the distribution"""
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1).reshape(-1, 1)
        self.cov = np.dot(data.T, data - self.mean) / (data.shape[0] - 1)
