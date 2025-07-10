#!/usr/bin/env python3
"""RNN Module"""
import numpy as np


class RNNCell:
    """Class that represents a simple RNN cell"""

    def __init__(self, i, h, o):
        """Constructor for RNNCell

        Args:
            i: the data
            h: the hidden state
            o: the outputs
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step

        Args:
            h_prev: previous hidden state
            x_t: data input for the cell

        Returns:
            h: new hidden state
            y: output of the cell
        """
        h_prev_x = np.concatenate((h_prev, x_t), axis=1)
        h = np.tanh(np.dot(h_prev_x, self.Wh) + self.bh)
        y = np.dot(h, self.Wy) + self.by
        return h, y