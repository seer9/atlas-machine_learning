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
    current = s
    for _ in range(t):
        next = np.matmul(current, P)
        current = next
    return current