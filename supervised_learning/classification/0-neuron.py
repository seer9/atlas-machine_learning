#!/usr/bin/env python3
import numpy as np
"""
this module containing the initialization if a single neuron

Classes:
    Neuron: the neuron class to initialize a single neuron
"""


class Neuron:
    """
    Represents a single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Initialize the neuron

        Parameters:
        nx (int): The number of input features to the neuron

        Raises:
        TypeError: If nx is not an integer
        ValueError: If nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")

        """initialize the weights, bias, and output of the neuron"""
        self.nx = nx
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
