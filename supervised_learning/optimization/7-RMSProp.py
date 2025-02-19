#!/usr/bin/env python3
"""RMSProp optimization algorithm"""
import tensorflow as tf


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    updates a variable using the RMSProp optimization algorithm
    """
    s = beta2 * s + (1 - beta2) * grad**2
    var = var - alpha * grad / (tf.sqrt(s) + epsilon)

    return var, s
