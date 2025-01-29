#!/usr/bin/env python3
"""module containing the neural network class"""
import numpy as np


class NeuralNetwork:
    """represention of a neural network"""

    def __init__(self, nx, nodes):
        """constructor method"""

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    def forward_prop(self, X):
        """calculates the forward propagation of the neural network"""

        Z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.W2, self.A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return (self.A1, self.A2)

    def cost(self, Y, A):
        """calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return (cost)

    def evaluate(self, X, Y):
        """evaluates the neural network's predictions"""
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.round(A2).astype(int)
        return (prediction, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """calculates one pass of gradient descent"""
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = np.matmul(A1, dz2.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        dz1 = A1 * (1 - A1) * np.matmul(self.W2.T, dz2)
        dw1 = np.matmul(X, dz1.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        self.__W2 = self.W2 - alpha * dw2.T
        self.__b2 = self.b2 - alpha * db2
        self.__W1 = self.W1 - alpha * dw1.T
        self.__b1 = self.b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """trains the neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        elif iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a number")
        elif alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            self.gradient_descent(X, Y, *self.forward_prop(X), alpha)

        return self.evaluate(X, Y)

    @property
    def W1(self):
        """private weights 1"""
        return self.__W1

    @property
    def b1(self):
        """private bias 1"""
        return self.__b1

    @property
    def A1(self):
        """private output 1"""
        return self.__A1

    @property
    def W2(self):
        """private weights 2"""
        return self.__W2

    @property
    def b2(self):
        """private bias 2"""
        return self.__b2

    @property
    def A2(self):
        """private output 2"""
        return self.__A2
