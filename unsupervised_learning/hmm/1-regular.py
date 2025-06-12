#!/usr/bin/env python3
import numpy as np


def regular(P):
    """
    Determines if a Markov chain is regular.

    Args:
        P is a is a square 2D numpy.ndarray of shape (n, n) representing the transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing the steady state probabilities, or None on failure
    """
    if not isinstance(P, np.ndarray):
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if len(P.shape) != 2:
        return None
    if not np.allclose(P.sum(axis=1), 1):
        return None

    n = P.shape[0]
    I = np.eye(n)
    A = np.subtract(I, P)
    A[-1, :] = 1
    b = np.zeros(n)
    b[-1] = 1
    try:
        steady_state = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    if np.any(steady_state < 0):
        return None
    return steady_state.reshape(1, n)