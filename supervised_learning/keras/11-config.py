#!/usr/bin/env python3
"""save/load config"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    network: model to save
    filename: path of the file that the model should be saved to
    """
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)
    return None


def load_config(filename):
    """
    filename: path of the file that the model should be loaded from
    """
    with open(filename, 'r') as f:
        config = f.read()
    network = K.models.model_from_json(config)
    return network
