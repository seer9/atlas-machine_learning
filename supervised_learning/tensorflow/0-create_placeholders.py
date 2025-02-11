#!/usr/bin/env python3
"""
placeholders
"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    param nx: number of feature columns in our data
    param classes: number of classes in our classifier
    return: placeholders named x and y, respectively
    """
    x = tf.placeholder("float32", shape=[None, nx], name="x")
    y = tf.placeholder("float32", shape=[None, classes], name="y")
    return (x, y)
