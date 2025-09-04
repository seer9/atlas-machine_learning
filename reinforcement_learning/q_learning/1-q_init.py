#!/usr/bin/env python3
"""initializing the Q-table"""
import numpy as np
import gymnasium as gym


def q_init(env):
    """initializes the Q-table.

    Args:
        env: the gym environment

    Returns:
        np.ndarray: the Q-table.
    """
    state_size = env.observation_space.n
    action_size = env.action_space.n

    q_table = np.zeros((state_size, action_size))

    return q_table
