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

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation

        Args:
            X_s: A numpy.ndarray of shape (m, d) containing the new input data.

        Returns:
            mu: represents the predicted mean.
            sigma: represents the predicted standard deviation.
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        mu_s = mu_s.reshape(-1)

        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma_s = np.diag(cov_s)

        return mu_s, sigma_s

    def update(self, X_new, Y_new):
        """
        Updates the Gaussian Process with new data.

        Args:
            X_new: A numpy.ndarray of shape (m, d) containing new input data.
            Y_new: A numpy.ndarray of shape (m,) containing new target values.
        """
        self.X = np.append(self.X, X_new, axis=0)
        self.Y = np.append(self.Y, Y_new, axis=0)
        self.K = self.kernel(self.X, self.X)
