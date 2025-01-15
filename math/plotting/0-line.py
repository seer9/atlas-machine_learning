#!/usr/bin/env python3
"""
in this module the function line is created using matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    creates a line graph.
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    """
    plots the line graph.
    """
    plt.plot(y, 'r')
    plt.xlim(0, 10)
    plt.show()
    return
