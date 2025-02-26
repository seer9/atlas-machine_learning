#!/usr/bin/env python3
"""l2 regularization gradient descent"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates weights and biases using gradient descent with L2 regularization
    Args:
        Y: one-hot that contains the correct labels for the data
        classes: classes
        m: data points
        weights: dict of weights and biases
        cache: dict of the outputs of each layer
        alpha: learning rate
        lambtha: L2 regularization parameter
        L: number of layers
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    """loops in reverse order"""
    for i in range(L, 0, -1):
        """previous layer"""
        A = cache['A' + str(i - 1)]
        """regularization"""
        l2_reg = (lambtha / m) * weights['W' + str(i)]
        """derivative of the cost with respect to Z"""
        dW = (1 / m) * np.matmul(dZ, A.T) + l2_reg
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            """derivative of the cost of A"""
            dA = 1 - A * A
            """derivative of the cost of Z"""
            dZ = np.matmul(weights['W' + str(i)].T, dZ) * dA

        """update weights and biases"""
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
