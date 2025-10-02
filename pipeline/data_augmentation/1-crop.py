#!/usr/bin/env python3
"""Crops an image"""
import tensorflow as tf


def crop_image(image, size):
    """Crops the center of an image.
    Args:
        image: the image to crop
        size: the size of the cropped image
    Returns: the cropped image
    """
    return tf.image.random_crop(image, size)
