#!/usr/bin/env python3
"""Adjusts the brightness of an image"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """Changes the brightness of an image.
    Args:
        image: image to change the brightness of
        max_delta: maximum delta for brightness adjustment
    Returns: brightness-adjusted image
    """
    return tf.image.random_brightness(image, max_delta)
