#!/usr/bin/env python3
"""prediction with keras"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    network: model making the prediction
    data: input data
    verbose: determines if output should be printed
    """
    prediction = network.predict(x=data, verbose=verbose)
    return prediction
