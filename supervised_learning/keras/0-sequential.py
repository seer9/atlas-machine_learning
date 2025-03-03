#!/usr/bin/env python3
"""Sequential model"""
import tensorflow.keras as keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx: number of inputs
    layers: nodes in each layer
    activations: activation functions used for each layer
    lambtha: L2 regularization
    keep_prob: probability that a node will be kept
    """
    model = keras.Sequential()
    for i, _ in enumerate(layers):
        model.add(keras.layers.Dense(
            layers[i],
            input_shape=(nx,),
            activation=activations[i],
            kernel_regularizer=keras.regularizers.l2(lambtha)
            ))
        if i < len(layers) - 1:
            model.add(
                keras.layers.Dropout(1 - keep_prob))
    return model