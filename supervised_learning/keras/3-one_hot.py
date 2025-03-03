#!/usr/bin/env python3
"""label to one-hot"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    The last dimension of the one-hot matrix
    must be the number of classes

    labels: labels of data
    classes: number of classes
    """
    one_hot = K.utils.to_categorical(labels, classes)
    return one_hot