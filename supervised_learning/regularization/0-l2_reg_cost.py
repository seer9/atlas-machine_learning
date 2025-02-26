#!/usr/bin/env python3
"""l2 regularization cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization
    Args:
        cost: cost without L2 regularization
        lambtha: regularization parameter
        weights: dictionary of the weights and biases
        L: number of layers 
        m: data points used
    Returns: cost of L2 regularization
    """
    l2_reg = 0
    
    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        l2_reg += np.sum(np.square(W))

    l2_reg *= (lambtha / (2 * m))
    
    return cost + l2_reg
