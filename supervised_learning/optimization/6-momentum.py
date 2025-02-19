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
    loss = tf.losses.get_total_loss()
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    train_op = optimizer.minimize(loss)
    return train_op
