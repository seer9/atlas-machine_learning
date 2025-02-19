#!/usr/bin/env python3
"""
sets up the gradient descent
with momentum optimization algorithm in TensorFlow
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    creates the training operation for a neural network in
    tensorflow using the gradient descent with momentum optimization algorithm
    """
    train_op = tf.keras.optimizers.SGD(
        learning_rate=alpha, momentum=beta1)
    return train_op
