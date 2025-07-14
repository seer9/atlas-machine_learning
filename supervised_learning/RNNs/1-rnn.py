#!/usr/bin/env python3
"""RNN Module"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN

    Args:
        rnn_cell: an instance of RNNCell that will be used for the forward pass
        X: contains the data to be used for the forward pass
        h_0: containing the initial hidden state

    Returns:
        H: all of the hidden states
        Y: all of the outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = []

    H[0] = h_0

    for step in range(t):
        H[step + 1], y = rnn_cell.forward(H[step], X[step])
        Y.append(y)

    Y = np.array(Y)
    return H, Y
