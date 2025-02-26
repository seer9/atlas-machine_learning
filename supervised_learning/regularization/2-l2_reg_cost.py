#!/usr/bin/env python3
"""l2 regularization cost"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    calculates the cost of a neural network with L2 regularization
    args:
    cost: a tensor containing the cost of the network without L2 regularization
    model: a Keras model that includes layers with L2 regularization.
    """
    return cost + model.losses
