#!/usr/bin/env python3
"""GRU Cell class"""
import numpy as np


class GRUCell:
    """Class that represents a Gated Recurrent cell"""

    def __init__(self, i, h, o):
        """initialize the GRU cell

        Args:
            i: the data
            h: the hidden state
            o: the outputs
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(x):
        """softmax activation function"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    @staticmethod
    def sigmoid(x):
        """sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        """perform forward propagation
        Args:
            h_prev: previous hidden state
            x_t: data input for the cell
        Returns:
            h: new hidden state
            y: output of the cell
        """
        h_prev_x = np.concatenate((h_prev, x_t), axis=1)

        z = self.sigmoid(np.dot(h_prev_x, self.Wz) + self.bz)
        r = self.sigmoid(np.dot(h_prev_x, self.Wr) + self.br)
        h_hat = np.tanh(np.dot(np.concatenate(
            (r * h_prev, x_t), axis=1), self.Wh) + self.bh)
        h = (1 - z) * h_prev + z * h_hat

        y_raw = np.dot(h, self.Wy) + self.by
        y = self.softmax(y_raw)

        return h, y
