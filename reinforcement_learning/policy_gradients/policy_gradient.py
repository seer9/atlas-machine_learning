#!/usr/bin/env python3
"""Policy Gradient"""
import numpy as np


def policy(matrix, weight):
    """computes the policy with a weight of a matrix.
    Args:
        matrix: the number of examples, n is the number of features.
        weight: where k is the number of classes.
    Returns: the softmax probabilities for each class, respectively.
    """
    z = matrix @ weight
    # compute softmax
    exp = np.exp(z)
    softmax = exp / np.sum(exp)
    return softmax


def policy_gradient(state, weight):
    """compute the gradient of the policy function with respect to its weights.
    Args:
        state: matrix representing the current observation of the environment.
        weight: matrix of random weights.
    Return: the action and the gradient (in this order).
    """
    # Reshape state to ensure it's a 2D array
    s = state.reshape(1, -1)
    # pass to policy
    pol = policy(s, weight)
    # now take action
    action = np.random.choice(len(pol[0]), p=pol[0])
    # compute gradient
    grad = s.T @ (np.eye(len(pol[0]))[action] - pol)
    return action, grad
