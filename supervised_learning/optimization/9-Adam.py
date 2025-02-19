#!/usr/bin/env python3
"""Adam optimization algorithm"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    updates a variable in place using the Adam optimization algorithm
    """
    v = beta1 * v + (1 - beta1) * grad
    new_v = v / (1 - beta1 ** t)
    s = beta2 * s + (1 - beta2) * grad ** 2
    new_s = s / (1 - beta2 ** t)
    var = var - alpha * new_v / (np.sqrt(new_s) + epsilon)

    return var, v, s
