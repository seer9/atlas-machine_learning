#!/usr/bin/env python3
"""learning rate decay"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """creates a learning rate decay operation in tensorflow using
    inverse time decay"""
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True)
