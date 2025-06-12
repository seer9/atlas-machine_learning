#!/usr/bin/env python3
"""Markov Chain"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Computes the state of a Markov chain after t iterations.

    Args:
        P: A numpy.ndarray of shape (n, n) representing the transition matrix.
        s: A numpy.ndarray of shape (n,) representing the initial state.
        t: The number of iterations to perform.

    Returns:
        A numpy.ndarray of shape (n,) representing the state after t iterations.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2 or P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or s.ndim != 1 or s.shape[0] != P.shape[0]:
        return None
    if not np.allclose(s.sum(), 1.0):
        return None

    current = s
    for _ in range(t):
        next = np.matmul(current, P)
        current = next
    return current