#!/usr/bin/env python3
"""Adam optimization algorithm"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow
    using the Adam optimization algorithm
    """
    train_op = tf.keras.optimizers.Adam(
        learning_rate=alpha, beta_1=beta1, beta_2=beta2, epsilon=epsilon)
    return train_op
