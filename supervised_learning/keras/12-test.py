#!/usr/bin/env python3
"""test the model"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    network: model to test
    data: input data
    labels: labels of data
    verbose: determines if output should be printed during the testing process
    """
    evaluation = network.evaluate(data, labels, verbose=verbose)
    return evaluation
