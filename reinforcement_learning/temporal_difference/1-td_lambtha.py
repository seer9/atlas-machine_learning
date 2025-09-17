#!/usr/bin/env python3
"""TD(λ) algorithm implementation"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """this function performs the TD(λ) algorithm to estimate state values
    Args:
        env: The environment instance
        V: containing the value estimate
        policy: takes in a state and returns the next action to take
        lambtha: the eligibility trace factor
        episodes: episodes to train over
        max_steps: number of steps per episode
        alpha: The learning rate
        gamma: The discount rate
    Returns:
        V: The updated value estimate.
    """
    # Loop over episodes
    for episode in range(episodes):
        # Reset the environment
        state = env.reset()[0]
        # Initialize eligibility traces
        E = np.zeros(V.shape)

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Calculate the TD error
            delta = reward + gamma * V[next_state] - V[state]

            # Update eligibility trace for the current state
            E[state] += 1

            # Update all state values and decay eligibility traces
            V += alpha * delta * E
            E *= gamma * lambtha

            if terminated or truncated:
                break
            state = next_state
    return V
