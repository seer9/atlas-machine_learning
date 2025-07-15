#!/usr/bin/env python3
"""Bidirectional RNN Cell"""
import numpy as np


class BidirectionalCell:
    """Bidirectional class"""
    def __init__(self, i, h, o):
        """Initialize the bidirectional cell

        Args:
            i: the data
            h: the hidden state
            o: the outputs
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation

        Args:
            h_prev: the previous hidden state
            x_t: the data input for the cell

        Returns:
            h_next: the next hidden state
        """
        prev_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(prev_x, self.Whf) + self.bhf)
        return h_next
