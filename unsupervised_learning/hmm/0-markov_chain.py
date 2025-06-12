#!/usr/bin/env python3
"""Markov Chain"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    computes the state of a Markov chain after t iterations.

    Args:
        P: represents the transition matrix.
        s: represents the initial state.
        t: The number of iterations to perform.

    Returns:
        the state after t iterations.
    """
    current = s
    for _ in range(t):
        next = np.matmul(current, P)
        current = next
    return current
