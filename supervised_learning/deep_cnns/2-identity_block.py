#!/usr/bin/env python3
"""function that builds an identity block"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described.

    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing F11, F3, F12.

    Returns:
        The activated output of the identity block.
    """
    [F11, F3, F12] = filters

    he_init = K.initializers.HeNormal(seed=0)

    # First 1x1 convolution
    X = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), padding='same',
                        kernel_initializer=he_init)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                        kernel_initializer=he_init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), padding='same',
                        kernel_initializer=he_init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
