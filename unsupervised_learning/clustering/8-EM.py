#!/usr/bin/env python3
"""EM algorithm"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    EM algorithm for GMM.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the data points.
        k: The number of clusters.
        iterations: The maximum number of iterations.
        tol: The tolerance for convergence.

    Returns:
        pi: A numpy.ndarray containing the cluster priors.
        m: A numpy.ndarray containing the cluster means.
        S: A numpy.ndarray containing the covariance matrices.
        g: A numpy.ndarray containing the posterior probabilities for each
           cluster and data point.
        log_likelihoods: A list of log likelihoods at each iteration.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol <= 0:
        return None, None, None, None, None

    pi, m, S = initualize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    prev_li = None
    for i in range(iterations):
    
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {round(prev_li, 5)}")

        pi, m, S = maximization(X, g)

        g, li = expectation(X, pi, m, S)

        if np.abs(li - prev_li) <= tol:
            break

    if verbose:
        print(f"Log Likelihood after {i + 1} iterations: {round(li, 5)}")
    return pi, m, S, g, li