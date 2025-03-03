#!/usr/bin/env python3
"""early stopping model"""
import tensorflow.keras as K


def train_model(
        network, data, labels, batch_size, epochs, validation_data=None,
        early_stopping=False, patience=0, verbose=True, shuffle=False):
    """
    network: model
    data: input data
    labels: labels of data
    batch_size: size of the batch
    epochs: number of passes through data
    validation_data: data to validate the model with
    early_stopping: boolean indicating whether early stopping should be used
    patience: patience used for early stopping
    verbose: determines if output should is printed during training
    shuffle: decides whether to shuffle the batches every epoch
    """
    callback = []

    if early_stopping and validation_data:
        callback = K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)
    History = network.fit(
        data, labels,
        batch_size=batch_size, epochs=epochs, verbose=verbose,
        shuffle=shuffle, validation_data=validation_data, callbacks=[callback])
    return History
