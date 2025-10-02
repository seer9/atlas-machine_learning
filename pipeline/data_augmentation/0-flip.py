#!/usr/bin/env python3
"""Flips an image horizontally"""
import tensorflow as tf


def flip_image(image):
    """Flips an image horizontally.
    Arg:
        image: the image to flip
    Returns: the flipped image
    """
    return tf.image.flip_left_right(image)
