#!/usr/bin/env python3
"""RMSProp optimization algorithm"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    updates a variable using the RMSProp optimization algorithm
    """
    new_s = beta2 * s + (1 - beta2) * np.power(grad, 2)
    new_var = var - alpha * grad / (np.sqrt(new_s) + epsilon)

    return new_var, new_s
