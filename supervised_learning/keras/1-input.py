#!/usr/bin/env python3
"""Input model"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx: number of inputs
    layers: nodes in each layer
    activations: activation functions used for each layer
    lambtha: L2 regularization
    keep_prob: probability that a node will be kept
    """
    inputs = K.Input(shape=(nx,))

    x = inputs
    for i, _ in enumerate(layers):
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
