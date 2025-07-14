#!/usr/bin/env python3
"""LSTM Cell class"""
import numpy as np


class LSTMCell():
    """Class that represents a Long Short Term Memory cell"""

    def __init__(self, i, h, o):
        """initialize the LSTM cell

        Args:
            i: the data
            h: the hidden state
            o: the outputs
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
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

    def forward(self, h_prev, c_prev, x_t):
        """perform forward propagation
        Args:
            h_prev: previous hidden state
            c_prev: previous cell state
            x_t: data input for the cell
        Returns:
            h_next: new hidden state
            c_next: new cell state
            y: output of the cell
        """
        h_prev_x = np.concatenate((h_prev, x_t), axis=1)

        # reset
        f = self.sigmoid(np.dot(h_prev_x, self.Wf) + self.bf)
        u = self.sigmoid(np.dot(h_prev_x, self.Wu) + self.bu)
        # update
        c_hat = np.tanh(np.dot(h_prev_x, self.Wc) + self.bc)
        c_next = f * c_prev + u * c_hat
        # output
        o = self.sigmoid(np.dot(h_prev_x, self.Wo) + self.bo)
        h_next = o * np.tanh(c_next)

        y_raw = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y_raw)

        return h_next, c_next, y
