#!/usr/bin/env python3
"""Q-learning implementation"""
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy
import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """trains an agent using the Q-learning algorithm.
    Args:
        env: The gym environment.
        Q: The Q-table.
        episodes: The total number of episodes to train over.
        max_steps: The maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount factor.
        epsilon: The initial epsilon for the
            epsilon-greedy policy.
        min_epsilon: The minimum value for epsilon.
        epsilon_decay: The decay rate for epsilon.
    Returns:
        Q: The updated Q-table.
        total_rewards: A list containing the rewards
            per episode.
    """
    total_rewards = []
    i_e = epsilon
    
    """initialize total rewards list"""
    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0

        """loop to take actions and update Q-table"""
        for step in range(max_steps):

            """calculate the action to take"""
            action = epsilon_greedy(Q, state, i_e)

            """perform the action and get the next state and reward"""
            new_state, reward, done, _, _ = env.step(action)
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action]
            )
            state = new_state
            total_reward += reward
            if done:
                break

        """decay epsilon"""
        i_e = max(min_epsilon, i_e - epsilon_decay)
        total_rewards.append(total_reward)
    return Q, total_rewards