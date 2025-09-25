#!/usr/bin/env python3
"""Policy Gradient"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Trains a policy gradient agent.
    Args:
        env: the initial environment
        nb_episodes: number of episodes used for training the agent
        alpha: learning rate
        gamma: discount factor
        show_result: if True, renders the environment every 1000 episodes
    Returns: all values of the score
    (sum of all rewards during one episode loop)
    """
    weight = np.random.rand(4, 2)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        state = state[None, :]
        score = 0
        done = False
        gradients = []
        rewards = []

        while not done and score < 500:
            if show_result and episode % 1000 == 0:
                env.render()
            action, grad = policy_gradient(state, weight)
            state, reward, done, _, _ = env.step(action)
            state = state[None, :]
            score += reward
            gradients.append(grad)
            rewards.append(reward)

        scores.append(score)

        # update weights
        for i in range(len(gradients)):
            weight += alpha * gradients[i] * sum(
                r * (gamma ** t) for t, r in enumerate(rewards[i:])
            )

        print(f"Episode:{episode} Score:{score}")

    return scores
