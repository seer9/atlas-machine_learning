#!/usr/bin/env python3
"""sarsa(lambda) algorithm implementation"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm to estimate the Q-table.

    Args:
        env: The environment instance.
        Q: The Q-table (numpy.ndarray) of shape (s, a).
        lambtha: The eligibility trace factor.
        episodes: The total number of episodes to train over.
        max_steps: The maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount rate.
        epsilon: The initial epsilon for the epsilon-greedy policy.
        min_epsilon: The minimum value of epsilon after decay.
        epsilon_decay: The decay rate for epsilon.

    Returns:
        Q: The updated Q-table.
    """
    init_epsilon = epsilon

    for episode in range(episodes):
        # reset the environment and initialize variables
        state = env.reset()[0]

        if np.random.uniform() < epsilon:
            action = np.random.randint(Q.shape[1])
        else:
            action = np.argmax(Q[state])

        # initialize eligibility trace
        E = np.zeros_like(Q)

        for _ in range(max_steps):
            # take action and observe reward and next state
            new_state, reward, terminated, truncated, _ = env.step(action)

            # choose next action using epsilon-greedy policy
            if np.random.uniform() < epsilon:
                new_action = np.random.randint(Q.shape[1])
            else:
                new_action = np.argmax(Q[new_state])

            # calculate TD error
            td_error = (
                reward + gamma * Q[new_state][new_action] - Q[state][action])

            # update eligibility trace
            E[state, action] += 1

            # ppdate Q-table and decay eligibility trace
            Q += alpha * td_error * E
            E *= gamma * lambtha

            # update state and action
            state, action = new_state, new_action

            if terminated or truncated:
                break

        # epsilon decay
        epsilon = min_epsilon + (
            init_epsilon - min_epsilon) * np.exp(-epsilon_decay * episode)

    return Q
