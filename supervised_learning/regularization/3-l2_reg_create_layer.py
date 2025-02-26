#!/usr/bin/env python3
"""l2 regularization cost"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    \tensorflow layer that includes L2 regularization
    args:
    prev: tensor containing the output of the previous layer
    n: nodes the new one should contain
    activation: activation function that should be used on the layer
    lambtha: l2 regularization
    """
    regularizer = tf.keras.regularizers.l2(lambtha)
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode=("fan_avg"))
    layer = tf.keras.layers.Dense(n, activation=activation,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer)
    return layer(prev)
