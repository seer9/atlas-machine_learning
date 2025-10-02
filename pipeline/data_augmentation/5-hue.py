#!/usr/bin/env python3
"""Adjusts the hue of an image"""
import tensorflow as tf


def change_hue(image, delta):
    """Changes the hue of an image.
    Args:
        image: image to change the hue of
        delta: maximum delta for hue adjustment
    Returns: hue-adjusted image
    """
    return tf.image.random_hue(image, delta)
