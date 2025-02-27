#!/usr/bin/env python3
"""dropout gradient descent"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    updates the weights w Dropout regularization
    :param Y: one-hot shape that contains the correct labels for the data
    :param weights: weights and biases 
    :param cache: outputs of each layer
    :param alpha: learning rate
    :param keep_prob: probability that a node will be kept
    :param L: number of layers
    """
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y

    for i in range(L, 0, -1):
        prevA = cache["A" + str(i - 1)]

        dW = (1 / m) * np.matmul(dZ, prevA.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dA = np.matmul(weights["W" + str(i)].T, dZ)
            dA *= cache["D" + str(i - 1)]
            dA /= keep_prob
            dZ = dA * (1 - (prevA ** 2))

        weights["W" + str(i)] -= alpha * dW
        weights["b" + str(i)] -= alpha * db
