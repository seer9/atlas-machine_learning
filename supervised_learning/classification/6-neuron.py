#!/usr/bin/env python3
"""this module containing the initialization if a single neuron"""
import numpy as np


class Neuron:

    """looks for a positive integer"""
    def __init__(self, nx):

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")

        """initialize the weights, bias, and output of the neuron"""
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """private weights"""
        return self.__W

    @property
    def b(self):
        """private bias"""
        return self.__b

    @property
    def A(self):
        """private output"""
        return self.__A

    def forward_prop(self, X):
        """the forward propagation of the neuron"""
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """the neuron's predictions"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = (A >= 0.5).astype(int)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """the gradient descent of the neuron"""
        m = Y.shape[1]
        dz = A - Y
        db = np.sum(dz) / m
        dw = np.matmul(X, dz.T) / m
        self.__b = self.__b - alpha * db
        self.__W = self.__W - alpha * dw.T

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """the training of the neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        elif not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a number")
        elif iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        elif not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a number")
        elif alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
        return self.evaluate(X, Y)
