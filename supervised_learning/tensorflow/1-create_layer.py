#!/usr/bin/env python3
"""
creating layers with tensorflow
"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    param prev: tensor output of the previous layer
    param n: number of nodes in the layer to create
    param activation: activation function that the layer should use
    return: tensor output of the layer
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0, mode="fan_avg", distribution="uniform")
    layer = tf.keras.layers.Dense(units=n, activation=activation,
        kernel_initializer=initializer, name='layer')
    return layer(prev)
