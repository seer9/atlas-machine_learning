#!/usr/bin/env python3
"""
this module contains the function change_scale
"""
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    creates a line graph with a logarithmic scale.
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))
    """
    plots the line graph.
    """
    plt.plot(x, y)
    plt.yscale('log')
    plt.xlim(0, 28650)
    plt.title('Exponential Decay of C-14')
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.show()
    return
