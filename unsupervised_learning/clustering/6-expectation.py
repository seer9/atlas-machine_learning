#!/usr/bin/env python3
"""EM for GMM"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for GMM.

    Args:
        X: A numpy.ndarray of shape (n, d) containing the data points.
        pi: A numpy.ndarray containing the cluster priors.
        m: A numpy.ndarray containing the cluster means.
        S: A numpy.ndarray containing the covariance matrices.

    Returns:
        A numpy.ndarray of shape (k, n) containing the posterior probabilities
        for each cluster and data point.
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(pi) is not np.ndarray or pi.ndim != 1:
        return None, None
    if type(m) is not np.ndarray or m.ndim != 2:
        return None, None
    if type(S) is not np.ndarray or S.ndim != 3:
        return None, None
    if X.shape[1] != m.shape[1] or m.shape[0] != S.shape[0]:
        return None, None
    if pi.shape[0] != m.shape[0] or pi.shape[0] != S.shape[0]:
        return None, None

    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    k = pi.shape[0]

    pdf_arr = np.array([pdf(X, m[i], S[i]) for i in range(k)])

    wpdf = pi[:, np.newaxis] * pdf_arr

    marg_prob = np.sum(wpdf, axis=0)
    post = wpdf / marg_prob

    likelihood = np.sum(np.log(marg_prob))

    return post, likelihood
