#!/usr/bin/env python3
"""
calculating accuracy of a prediction
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    peram y: placeholder for the labels of the input data
    peram y_pred: a tensor containing the networks predictions
    returns: a tensor containing the prediction in decimal accuracy
    """
    accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    mean = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    return mean
