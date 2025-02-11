#!/usr/bin/env python3
"""
training operation for the network
"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    peram loss: the loss of the networks prediction
    peram alpha: the learning rate
    returns: the operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op
