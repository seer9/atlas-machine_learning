#!/usr/bin/env python3
"""Input model"""
import tensorflow.keras as keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx: number of inputs
    layers: nodes in each layer
    activations: activation functions used for each layer
    lambtha: L2 regularization
    keep_prob: probability that a node will be kept
    """
    inputs = keras.Input(shape=(nx,))

    x = inputs
    for i, _ in enumerate(layers):
        x = keras.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=keras.regularizers.l2(lambtha)
        )(x)
        if i < len(layers) - 1:
            x = keras.layers.Dropout(1 - keep_prob)(x)
    
    model = keras.Model(inputs=inputs, outputs=x)
    return model
