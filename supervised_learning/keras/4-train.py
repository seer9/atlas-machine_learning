#!/usr/bin/env python3
"""training/fitting a model"""
import tensorflow.keras as K


def train_model(
        network, data, labels, batch_size,
        epochs, verbose=True, shuffle=False):
    """
    network: model
    data: input data
    labels: labels of data
    batch_size: size of the batch
    epochs: number of passes through data
    verbose: determines if output should is printed during training
    shuffle: decides whether to shuffle the batches every epoch
    """
    History = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )
    return History
