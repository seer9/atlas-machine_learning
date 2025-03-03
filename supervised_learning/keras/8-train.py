#!/usr/bin/env python3
"""learning rate decay model"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    network: model
    data: input data
    labels: labels of data
    batch_size: size of the batch
    epochs: number of passes through data
    validation_data: data to validate the model
    early_stopping: determines the early stopping that is used
    patience: patience used for early stopping
    learning_rate_decay: determines the learning rate decay thats used
    alpha: learning rate
    decay_rate: decay rate
    verbose: determines if output should is printed during training
    shuffle: decides whether to shuffle the batches every epoch
    """
    callback = []

    if early_stopping and validation_data:
        callback = K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)

    if learning_rate_decay and validation_data:
        def scheduler(epoch):
            """
            for callback, calculates learning rate
            """
            return (alpha / (1 + decay_rate * epoch))
        callback.append(K.callbacks.LearningRateScheduler(
            scheduler, verbose=1))

    if save_best and filepath:
        callback.append(K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            mode='min',
            save_best_only=True))

    History = network.fit(
        data, labels,
        batch_size=batch_size, epochs=epochs, verbose=verbose,
        shuffle=shuffle, validation_data=validation_data, callbacks=callback)
    return History
