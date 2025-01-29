#!/usr/bin/env python3
"""module containing the deep neural network class"""
import numpy as np


class DeepNeuralNetwork:
    """the deep neural network class"""

    def __init__(self, nx, layers):
        """constructor module"""

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        """the layers of the network"""
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        """weights and biases of the network"""
        for i in range(1, self.L + 1):

            size = layers[i - 1]
            prev = nx if i == 1 else layers[i - 2]

            self.weights[f'W{i}'] = (
                  np.random.randn(size, prev) * np.sqrt(2 / prev))
            self.weights[f'b{i}'] = np.zeros((size, 1))
