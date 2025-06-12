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
    if not isinstance(P, np.ndarray) or P.shape[0] != P.shape[1]:
        return None
    if len(P.shape) != 2:
        return None
    if not np.allclose(P.sum(axis=1), 1):
        return None

    # setting up the equation
    n = P.shape[0]
    I = np.eye(n)
    A = np.transpose(P) - I
    b = np.zeros((n, 1))
    b[-1] = 1  # last row is 1 for the steady state equation
    # Solve the linear system
    try:
        state = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return state.flatten() / state.sum()  # Normalize to get probabilities
