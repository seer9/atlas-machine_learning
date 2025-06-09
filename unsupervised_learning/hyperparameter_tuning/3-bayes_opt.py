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
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Initializes the Bayesian Optimization with the function to optimize, initial data, and parameters.
        
        Args:
            f: The black-box function to be optimized.
            X_init: representing the initial input data points.
            Y_init: representing the initial target values.
            bounds: representing the bounds of the space to search.
            ac_samples: The number of samples to consider for the acquisition function.
            l: The length-scale parameter for the Gaussian Process kernel.
            sigma_f: The signal variance for the Gaussian Process kernel.
            xsi: The exploration-exploitation trade-off parameter for the acquisition function.
            minimize: indicator whether to minimize (True) or maximize (False) the objective function.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        X_s = np.linspace(bounds[0], bounds[1], ac_samples)
        self.X_s = X_s.reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
