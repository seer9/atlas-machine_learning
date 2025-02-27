#!/usr/bin/env python3
"""dropout regularization"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    creates a layer using dropout
    :param prev: tensor of the previous layer
    :param n: number of nodes to the new layer
    :param activation: activation function used on the layer
    :param keep_prob: chance that a node is kept
    :param training: boolean to when the model is training
    :return: the new layer
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode=("fan_avg"))
    layer = tf.keras.layers.Dense(
        units=n, activation=activation, kernel_initializer=initializer)
    out = layer(prev)

    if training:
        drop = tf.keras.layers.Dropout(rate=(1 - keep_prob))
        output = drop(out, training=training)
    return output
