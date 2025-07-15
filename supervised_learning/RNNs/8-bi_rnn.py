#!/usr/bin/env python3
"""
Bidirectional RNN forward propagation module.
IMPORTS ALREADY IN MAIN
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Perform forward propagation for a bidirectional RNN.

    Args:
        bi_cell: BidirectionalCell instance.
        X: the input data
        h_0: the initial hidden state
        h_t: the initial backward hidden state

    Returns:
        H: the concatenated hidden states
        Y: the outputs
    """
    t, m, _ = X.shape
    h = h_0.shape[1]
    Hf = np.zeros((t, m, h))
    Hb = np.zeros((t, m, h))
    h_next, h_prev = h_0, h_t

    for time_step in range(t):
        h_next = bi_cell.forward(h_next, X[time_step])
        Hf[time_step] = h_next
        h_prev = bi_cell.backward(h_prev, X[t - time_step - 1])
        Hb[t - time_step - 1] = h_prev

    H = np.concatenate((Hf, Hb), axis=2)
    Y = bi_cell.output(H)
    return H, Y
