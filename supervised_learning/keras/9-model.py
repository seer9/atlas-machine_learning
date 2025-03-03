#!/usr/bin/env python3
"""save/load model"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    network: model to save
    filename: path of the file that the model should be saved to
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    filename: path of the file that the model should be loaded from
    """
    network = K.models.load_model(filename)
    return network
