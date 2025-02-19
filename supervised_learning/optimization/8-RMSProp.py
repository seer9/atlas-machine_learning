#!/usr/bin/env python3
"""RMSProp optimization algorithm"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    training operation for a neural network using the
    RMSProp optimization algorithm
    """
    train_op = tf.keras.optimizers.RMSprop(
        learning_rate=alpha, rho=beta2, epsilon=epsilon)
    return train_op
