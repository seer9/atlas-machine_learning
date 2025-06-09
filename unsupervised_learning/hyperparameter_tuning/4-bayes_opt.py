#!/usr/bin/env python3
"""Bayesian Optimization with Gaussian Process"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Class for Bayesian Optimization using Gaussian Process.

    Attributes:
        gp: An instance of GaussianProcess.
        X_s: A numpy.ndarray of shape (m, d) representing the points to sample.
    """
    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Bayesian Optimization to optimize, initial data, and parameters.

        Args:
            f: The black-box function to be optimized.
            X_init: representing the initial input data points.
            Y_init: representing the initial target values.
            bounds: representing the bounds of the space to search.
            ac_samples: samples to consider for the acquisition function.
            l: The length-scale parameter for the Gaussian Process kernel.
            sigma_f: The signal variance for the Gaussian Process kernel.
            xsi: trade-off parameter for the acquisition function.
            minimize: indicator whether to minimize or maximize the objective.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        X_s = np.linspace(bounds[0], bounds[1], ac_samples)
        self.X_s = X_s.reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Computes the acquisition function based on the Gaussian Process.

        Returns:
            X_next: representing the next best sample point.
            EI: containing the expected improvement.
        """
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize is True:
            Y_sampled = np.min(self.gp.Y)
        else:
            Y_sampled = np.max(self.gp.Y)
            imp = mu - Y_sampled - self.xsi
        
        Z = np.zeros(sigma.shape[0])
        for i in range(sigma.shape[0]):
            if sigma[i] == 0:
                Z[i] = 0
            else:
                Z[i] = imp[i] / sigma[i]
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI
