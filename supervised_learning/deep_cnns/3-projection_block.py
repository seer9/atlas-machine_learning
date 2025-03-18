#!/usr/bin/env python3
"""function that builds a projection block"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s):
    """
    Builds a projection block as described.

    Args:
        A_prev: output from the previous layer
        filters: tuple or list containing F11, F3, F12.

    Returns:
        The activated output of the projection block.
    """
    F11, F3, F12 = filters

    he_init = K.initializers.HeNormal(seed=0)

    X = K.layers.Conv2D(
        filters=F11, kernel_size=(1, 1),
        strides=s, padding='same', kernel_initializer=he_init)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        filters=F3, kernel_size=(3, 3),
        strides=1, padding='same', kernel_initializer=he_init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(
        filters=F12, kernel_size=(1, 1),
        strides=1, padding='same', kernel_initializer=he_init)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    shortcut = K.layers.Conv2D(
        filters=F12, kernel_size=(1, 1),
        strides=s, padding='same', kernel_initializer=he_init)(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    X = K.layers.Add()([X, shortcut])
    X = K.layers.Activation('relu')(X)

    return X
