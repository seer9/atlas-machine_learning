#!/usr/bin/env python3
"""variational autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder with the specified dimensions.

    Args:
        input_dims: input data dimensions
        hidden_layers: number of units in each hidden layer
        latent_dims: the latent space dimensions

    Returns:
        tuple: Encoder, decoder, and autoencoder models.
    """
    # Input layer
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    # Hidden layers with relu activation
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    # Latent space parameters
    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)  # Mean of the latent space
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)  # Log variance of the latent space

    # Sampling layer
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = keras.backend.random_normal(shape=keras.backend.shape(z_mean))
        return z_mean + keras.backend.exp(z_log_var / 2) * epsilon

    # Latent representation
    latent = keras.layers.Lambda(sampling)([z_mean, z_log_var])

    # Encoder model
    encoder = keras.Model(inputs, [latent, z_mean, z_log_var], name='encoder')

    # Decoder model
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # Autoencoder model
    autoencoder = keras.Model(inputs, decoder(encoder(inputs)[0]), name='autoencoder')

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
