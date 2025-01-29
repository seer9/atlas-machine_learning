#!/usr/bin/env python3
import numpy as np
"""this module containing the initialization if a single neuron"""


class Neuron:

    """set it to look for a positive integer"""
    def __init__(self, nx):

        """looks for a positive integer"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")

        """initialize the weights, bias, and output of the neuron"""
        self.nx = nx
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
