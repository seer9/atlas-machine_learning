#!/usr/bin/env python3
"""
in this module, there is a function called bars()
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    this function creates a bar graph.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))
    """
    plotting the bar graph.
    """
    labels = ['Farrah', 'Fred', 'Felicia']
    width = 0.5
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    labels = ['apples', 'bananas', 'oranges', 'peaches']
    bottom = np.zeros(3)
    for i in range(4):
        plt.bar(['Farrah', 'Fred', 'Felicia'], fruit[i], bottom=bottom,
                color=colors[i], label=labels[i], width=width)
        bottom += fruit[i]
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.ylim(0, 80)
    plt.show()
    return
