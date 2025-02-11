#!/usr/bin/env python3
"""
forward propagation
"""
import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    param x: placeholder
    param layer_sizes: list with the number of nodes in each layer
    param activations: list of the activation functions for each layer
    return: prediction of the network in tensor form
    """
    create_layer = __import__('1-create_layer').create_layer
    layer = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[i], activations[i])
    return layer
