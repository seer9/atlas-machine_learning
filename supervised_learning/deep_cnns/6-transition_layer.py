#!/usr/bin/env python3
"""function for a transition layer"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    builds the transition layer.
    Args:
        X: output from the previous layer
        nb_filters: integer representing the number of filters in X
        compression: compression factor for the transition layer

    Returns:
        the output of the transition layer and the number of filters.
    """
    he_init = K.initializers.he_normal()
    batch_norm = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation('relu')(batch_norm)
    nb_filters = int(nb_filters * compression)
    conv = K.layers.Conv2D(nb_filters,
                           kernel_size=1,
                           padding='same',
                           kernel_initializer=he_init)(activation)
    avg_pool = K.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2))(conv)
    return (avg_pool, nb_filters)
