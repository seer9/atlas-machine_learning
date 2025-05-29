#!/usr/bin/env python3
"""BIC for GMM"""
import numpy as np
expectation_max = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using BIC.

    Args:
        X: containing the data set.
        kmin: minimum number of clusters to check for (inclusive).
        kmax: maximum number of clusters to check for (inclusive).
        iterations: maximum number of iterations for the EM algorithm.
        tol: the tolerance for the EM algorithm.
        verbose: A boolean for the EM algorithm should print information to
        the standard output.

    Returns:
        best_k: The best value for k
        best_result: the best number of clusters.
        l: the log likelihood for each cluster size tested.
        b: the BIC value for each cluster size tested.
        Returns None, None, None, None on failure.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax < kmin):
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    log_likelihoods = []
    BICs = []
    best_result = None
    best_k = None

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_max(X, k,
                                                      iterations=iterations,
                                                      tol=tol,
                                                      verbose=verbose)
        if pi is None or m is None:
            return None, None, None, None
        if S is None or g is None:
            return None, None, None, None

        num_params = (k - 1) + (k * d) + (k * d * (d + 1)) / 2

        BIC = np.log(n) * num_params - 2 * log_likelihood

        log_likelihoods.append(log_likelihood)
        BICs.append(BIC)

        if best_k is None or BIC < BICs[best_k - kmin]:
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, np.array(log_likelihoods), np.array(BICs)
