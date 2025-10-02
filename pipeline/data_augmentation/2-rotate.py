#!/usr/bin/env python3
"""Rotates an image"""
import tensorflow as tf


def rotate_image(image):
    """Rotates an image 90 degrees counter-clockwise.
    Args:
        image: the image to rotate
    Returns: the rotated image
    """
    return tf.image.rot90(image)
