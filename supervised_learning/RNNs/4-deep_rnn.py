#!/usr/bin/env python3
"""Deep RNN module"""
import numpy as np


def deep_rnn(rnn_cells, X, H_0):
    """Performs forward propagation for a deep RNN

    Args:
        rnn_cells: list of RNNCell instances
        X: numpy.ndarray of shape (t, m, i) containing the data to be used
            in the forward propagation
        H_0: numpy.ndarray of shape (l, m, h) containing the initial hidden
            states for each layer

    Returns:
        H: numpy.ndarray containing all of the hidden states
        Y: numpy.ndarray containing all of the outputs
    """
    LAYERS = len(rnn_cells)
    H, Y = [H_0], []
    temp_H = H_0.copy()

    for x in X:
        temp_H[0], _ = rnn_cells[0].forward(temp_H[0], x)
        for l in range(LAYERS - 1):
            temp_H[l + 1], _ = rnn_cells[l + 1].forward(temp_H[l+1], temp_H[l])
            if l == LAYERS - 2:
                Y.append(0)
        H.append(temp_H.copy())
    return np.array(H), np.array(Y)
