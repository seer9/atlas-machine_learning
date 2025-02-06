#!/usr/bin/env python3
"""module containing the deep neural network class"""
import numpy as np


class DeepNeuralNetwork:
    """a deep neural network"""

    def __init__(self, nx, layers):
        """constructor method"""
        if not isinstance(nx, int) or nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0 or not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(1, self.__L + 1):
            size = layers[i - 1]
            prev = nx if i == 1 else layers[i - 2]

            self.__weights[f'W{i}'] = np.random.randn(size, prev) * np.sqrt(2 / prev)
            self.__weights[f'b{i}'] = np.zeros((size, 1))

    def forward_prop(self, X):
        """Forward propagation of the network"""
        self.__cache['A0'] = X

        for i in range(1, self.L + 1):
            W = self.weights[f'W{i}']
            b = self.weights[f'b{i}']
            A = self.cache[f'A{i - 1}']
            Z = np.matmul(W, A) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache[f'A{i}'] = A

        return A, self.cache

    def cost(self, Y, A):
        """the cost of the network"""
        m = Y.shape[1]
        return (
            -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m)

    def evaluate(self, X, Y):
        """evaluate the network"""
        A, _ = self.forward_prop(X)
        return np.round(A).astype(int), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """gradient descent in the network"""
        m = Y.shape[1]
        dZ = self.cache[f'A{self.L}'] - Y

        for i in range(self.L, 0, -1):
            A = self.cache[f'A{i - 1}']
            W = self.weights[f'W{i}']
            b = self.weights[f'b{i}']

            dW = np.matmul(dZ, A.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dZ = np.matmul(W.T, dZ) * A * (1 - A)

            self.__weights[f'W{i}'] = self.weights[f'W{i}'] - alpha * dW
            self.__weights[f'b{i}'] = self.weights[f'b{i}'] - alpha * db

    @property
    def L(self):
        """Getter method L"""
        return self.__L

    @property
    def cache(self):
        """Getter method cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter method weights"""
        return self.__weights
