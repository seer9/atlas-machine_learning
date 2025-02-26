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
        A = cache['A' - str(i - 1)]
        Z = np.matmul(weights['W' + str(i + 1)], A) + weights['b' + str(i)]
        
        if i < L:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A = np.multiply(A, D)
            A /= keep_prob
            cache['D' + str(i + 1)] = D
        else:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)

        cache['A' + str(i + 1)] = A
    
    return (cache)
