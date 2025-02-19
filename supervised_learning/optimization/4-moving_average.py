#!/usr/bin/env python3
"""moving average"""


def moving_average(data, beta):
    """
    calculates the weighted moving average of a data set
    :param data: list of data to calculate the moving average of
    :param beta: weight used for the moving average
    :return: list containing the moving averages of data
    """
    moving_averages = []
    v = 0
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        moving_averages.append(v / (1 - beta ** (i + 1)))
    return moving_averages
