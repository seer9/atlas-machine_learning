#!/usr/bin/env python3
"""dropout forward prop"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    conducts forward propagation using Dropout
    :param X: np.ndarray (nx, m) of input data
    :param weights: dictionary of weights and biases of the neural network
    :param L: number of layers in the network
    :param keep_prob: probability that a node will be kept
    :return: dictionary containing the output of each layer and
    the dropout mask used on each layer
    """
    cache = {}
    cache = {"A0": X}

    for i in range(1, L + 1):
        prev_A = cache[f"A{i - 1}"]
        Z = np.matmul(weights[f"W{i}"], prev_A) + weights[f"b{i}"]

        if i < L:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            cache[f"D{i}"] = D
            A = np.multiply(A, D)
            A /= keep_prob
        else:
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

        cache['A' + str(i)] = A

    return (cache)
