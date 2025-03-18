#!/usr/bin/env python3
"""builds a dense block in keras"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    builds a dense block using bottleneck layers for DenseNet-B.

    args:
        X: the output from the previous layer.
        nb_filters: integer representing the number of filters in X.
        growth_rate: growth rate for the dense block.
        layers: number of layers in the dense block.

    Returns:
        The concatenated output of each layer within the Dense Block and the
        number of filters within the concatenated outputs, respectively.
    """
    he_init = K.initializers.HeNormal(seed=0)

    for _ in range(layers):
        x = K.layers.BatchNormalization()(X)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(
            filters=4 * growth_rate, kernel_size=1, padding='same',
            kernel_initializer=he_init)(x)

        x = K.layers.BatchNormalization()(x)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(
            filters=growth_rate, kernel_size=3, padding='same',
            strides=1, kernel_initializer=he_init)(x)

        X = K.layers.Concatenate()([X, x])
        nb_filters += growth_rate

    return (X, nb_filters)
