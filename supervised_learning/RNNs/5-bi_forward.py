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
    def sigmoid(x):
        """sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def forward(self, h_prev, x_t):
        """Perform forward propagation

        Args:
            h_prev: previous hidden state
            x_t: data input for the cell

        Returns:
            h_next: new hidden state
        """
        h_prev_x = np.concatenate((h_prev, x_t), axis=1)

        # forward
        h_forward = self.sigmoid(np.dot(h_prev_x, self.Whf) + self.bhf)

        # backward
        h_backward = self.sigmoid(np.dot(h_prev_x, self.Whb) + self.bhb)

        # concatenate both
        h_next = np.concatenate((h_forward, h_backward), axis=1)

        return h_next
