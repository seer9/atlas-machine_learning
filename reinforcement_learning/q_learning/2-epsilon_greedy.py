#!/usr/bin/env python3
"""epsilon-greed action selection"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """selects an action using the epsilon-greedy
    policy.
    Args:
        Q: The Q-table.
        state: The current state.
        epsilon: The epsilon to use for the
            epsilon-greedy policy.
    """
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state])

    return action
