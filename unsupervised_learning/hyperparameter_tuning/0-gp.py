#!/usr/bin/env python3
"""Gaussian Process"""
import numpy as np


class GaussianProcess:
    """
    Represents a Gaussian Process for hyperparameter tuning.

    Attributes:
        X: A numpy.ndarray of shape (n, d) containing the input data.
        Y: A numpy.ndarray of shape (n,) containing the target values.
        kernel: A callable kernel function for covariance.
        noise: The noise level in the observations.
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        initializes the Gaussian Process with initial data and parameters.

        Args:
            X_init: shape (n, d)
            Y_init: shape (n,)
            l: is the length parameter for the kernel
            sigma_f: Signal variance for the kernel
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Computes the covariance matrix using the squared exponential kernel.

        Args:
            X1: First input data of shape (n, d).
            X2: Second input data of shape (m, d).

        Returns:
            A numpy.ndarray of shape (n, m) representing the covariance matrix.
        """
        sqdist1 = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1)
        sqdist1 -= 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist1)
