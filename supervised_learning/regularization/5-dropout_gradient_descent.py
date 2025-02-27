#!/usr/bin/env python3
"""dropout gradient descent"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    updates the weights of a neural network with Dropout regularization
    :param Y: one-hot np.ndarray of shape (classes, m) that contains the
    correct labels for the data
    :param weights: dictionary of weights and biases of the neural network
    :param cache: dictionary of outputs of each layer of the neural network
    :param alpha: learning rate
    :param keep_prob: probability that a node will be kept
    :param L: number of layers of the network
    :return: None
    """
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache["A" + str(i - 1)]
        W = weights["W" + str(i)]
        dW = np.matmul(dZ, A.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.matmul(W.T, dZ)
        if i > 1:
            dA = dA * cache["D" + str(i - 1)]
            dA = dA / keep_prob
            dA = dA * (1 - A ** 2)
        dZ = dA
        weights["W" + str(i)] = weights["W" + str(i)] - alpha * dW
        weights["b" + str(i)] = weights["b" + str(i)] - alpha * db