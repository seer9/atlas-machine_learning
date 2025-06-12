#!/usr/bin/env python3
import numpy as np


def regular(P):
    """
    Determines if a Markov chain is regular.

    Args:
        P is shape (n, n) representing the transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n is the number of states in the markov chain
    Returns: the steady state probabilities, or None on failure
    """
    try:
        if not isinstance(P, np.ndarray):
            return None
        if P.shape[0] != P.shape[1]:
            return None
        if len(P.shape) != 2:
            return None
        if not np.allclose(P.sum(axis=1), 1):
            return None

        n = P.shape[0]
        evals, evecs = np.linalg.eig(P.T)
        # normalize
        state = (evecs / evecs.sum())

        new_S = np.dot(state.T, P)
        for i in new_S:
            if (i >= 0).all() and np.isclose(i.sum(), 1):
                return i.reshape(1, n)
    except Exception:
        return None
