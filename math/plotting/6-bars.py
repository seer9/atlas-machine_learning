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
    plt.bar(labels, fruit[0], width, color='r', label='apples')
    plt.bar(labels, fruit[1], width, color='yellow', label='bananas', bottom=fruit[0])
    plt.bar(labels, fruit[2], width, color='#ff8000', label='oranges', bottom=fruit[0]+fruit[1])
    plt.bar(labels, fruit[3], width, color='#ffe5b4', label='peaches', bottom=fruit[0]+fruit[1]+fruit[2])
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.ylim(0, 80)
    plt.show()
    return
