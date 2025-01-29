#!/usr/bin/env python3
"""this module containing the initialization if a single neuron"""
import numpy as np


class Neuron:

    """set it to look for a positive integer"""
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
