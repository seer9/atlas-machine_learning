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

    @staticmethod
    def softmax(x):
        """softmax activation function

        Args:
            x: input data

        Returns:
            softmax output
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

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

    def backward(self, h_next, x_t):
        """Perform backward propagation

        Args:
            h_next: the next hidden state
            x_t: the data input for the cell

        Returns:
            h_prev: the previous hidden state
        """
        next_x = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(next_x, self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """Calculate the output of the bidirectional cell

        Args:
            H: the concatenated hidden states from both directions

        Returns:
            y: the output of the cell
        """
        y_raw = np.dot(H, self.Wy) + self.by
        y = self.softmax(y_raw)
        return y
