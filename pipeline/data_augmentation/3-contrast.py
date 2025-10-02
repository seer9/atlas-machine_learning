#!/usr/bin/env python3
"""Adjusts the contrast of an image"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """Changes the contrast of an image.
    Args:
        image: image to change the contrast of
        lower: lower bound of the contrast factor
        upper: upper bound
    Returns: contrast-adjusted image
    """
    cf = tf.random.uniform([], lower, upper)

    return tf.image.adjust_contrast(image, cf)
