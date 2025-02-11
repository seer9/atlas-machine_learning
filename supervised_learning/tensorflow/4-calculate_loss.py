#!/usr/bin/env python3
"""
calculating the softmax cross-entropy loss of a prediction
"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    peram y: placeholder for the labels of the input data
    peram y_pred: a tensor containing the networks predictions
    returns: tensor containing the loss
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
