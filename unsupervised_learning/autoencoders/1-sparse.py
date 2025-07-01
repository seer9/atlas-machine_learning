#!/usr/bin/env python3
"""sparse autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates a sparse autoencoder with the specified dimensions.

    Args:
        input_dims: input data
        hidden_layers: number of units in each hidden layer
        latent_dims: the latent space
        lambtha: regularization parameter

    Returns:
        tuple: Encoder, decoder, and autoencoder models.
    """
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='relu')(x)

    encoder = keras.Model(inputs, latent)

    x = latent
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(latent, outputs, name='decoder')

    autoencoder = keras.Model(
        inputs, decoder(encoder(inputs)), name='autoencoder')

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy',
                        metrics=[keras.metrics.MeanAbsoluteError()],
                        loss_weights={'decoder': 1.0,
                                      'encoder': lambtha})

    return encoder, decoder, autoencoder
