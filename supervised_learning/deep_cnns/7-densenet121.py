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
    he_init = K.initializers.he_normal(seed=None)

    # Initial Convolution
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(
        filters=64, kernel_size=7, padding='same',
        strides=2, kernel_initializer=he_init)(X)
    X = K.layers.MaxPooling2D(
        pool_size=3, strides=2, padding='same')(X)

    # Dense Block 1
    X, nb_filters = dense_block(X, 64, growth_rate, 6)
    # Transition Layer 1
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    # Transition Layer 2
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    # Transition Layer 3
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Global Average Pooling
    X = K.layers.AveragePooling2D(
        pool_size=7, padding='same')(X)

    # Fully Connected Softmax Layer
    X = K.layers.Dense(
        units=1000, activation='softmax',
        kernel_initializer=he_init)(X)

    model = K.models.Model(inputs=X, outputs=X)

    return model
