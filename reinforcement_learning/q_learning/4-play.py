#!/usr/bin/env python3
"""training an agent to play FrozenLake"""
import numpy as np


def play(env, Q, max_steps=100):
    """trains an agent to play FrozenLake
    using the Q-learning algorithm.

    Args:
        env: The FrozenLake environment.
        Q: The Q-table.
        max_steps: The maximum number of steps per episode.

    Returns:
        total_rewards: The total rewards for the episode.
        rendered_outputs: A list of strings containing the
            rendered outputs of each step.
    """
    """initializzing starting state"""
    state = env.reset()[0]
    done = False
    total_rewards = 0
    outputs = []

    for _ in range(max_steps):
        """rendering the environment"""
        outputs.append(env.render())

        """choosing the action with the highest Q-value"""
        action = np.argmax(Q[state])

        """taking the action and observing the next state and reward"""
        new_state, reward, done, _, _ = env.step(action)

        """updating the total rewards"""
        total_rewards += reward

        """updating the state"""
        state = new_state

        """checking if the episode is done"""
        if done:
            break

        """final rendering of the environment"""
        outputs.append(env.render())
    
    return total_rewards, outputs

