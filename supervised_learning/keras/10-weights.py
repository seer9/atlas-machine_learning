#!/usr/bin/env python3
"""save and load weights"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    network: weights that get be saved
    filename: path of the file
    save_format: format in which the weights should be saved
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    network: weights that get be loaded
    filename: path of the file
    """
    network.load_weights(filename)
    return None
