#!/usr/bin/env python3
"""builds the DenseNet-121 architecture"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    builds the DenseNet-121 architecture.

    args:
        growth_rate: growth rate for the dense block.
        compression: compression factor for the transition layer.

    Returns:
        The keras model.
    """
    X = K.Input(shape=(224, 224, 3))
    he_init = K.initializers.HeNormal(seed=0)

    x = K.layers.BatchNormalization()(X)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(
        filters=64, kernel_size=7, strides=2,
        padding='same', kernel_initializer=he_init)(x)
    x = K.layers.MaxPooling2D(
        pool_size=3, strides=2, padding='same')(x)

    x, nb_filters = dense_block(x, 64, growth_rate, 6)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 12)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 24)
    x, nb_filters = transition_layer(x, nb_filters, compression)

    x, nb_filters = dense_block(x, nb_filters, growth_rate, 16)

    x = K.layers.AveragePooling2D(pool_size=7, padding='same')(x)
    x = K.layers.Dense(1000, activation='softmax')(x)

    model = K.models.Model(inputs=X, outputs=x)

    return model
