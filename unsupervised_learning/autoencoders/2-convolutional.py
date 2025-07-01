#!/usr/bin/env python3
"""convolutional autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder.

    Args:
        input_dims: input data dimensions
        filters: number of filters
        latent_dims: the latent space dimensions

    Returns:
        tuple: Encoder, decoder, and autoencoder models.
    """
    inputs = keras.Input(shape=input_dims)
    x = inputs
    for f in filters:
        x = keras.layers.Conv2D(
            f, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Latent layer
    latent = keras.layers.Conv2D(
        latent_dims[2],
        kernel_size=(3, 3),
        activation='relu',
        padding='same')(x)

    encoder = keras.Model(inputs, latent)

    # Decoder
    x = latent
    for i, f in enumerate(reversed(filters)):
        if i < len(filters) - 1:  # All but the second-to-last and last layers
            x = keras.layers.Conv2DTranspose(
                f, kernel_size=(3, 3), activation='relu', padding='same')(x)
            x = keras.layers.UpSampling2D(size=(2, 2))(x)
        elif i == len(filters) - 1:  # Second-to-last layer
            x = keras.layers.Conv2DTranspose(
                f, kernel_size=(3, 3), activation='relu', padding='valid')(x)

    # Last layer
    out = keras.layers.Conv2D(
        input_dims[2],
        kernel_size=(3, 3),
        activation='sigmoid',
        padding='same')(x)

    decoder = keras.Model(latent, out, name='decoder')

    # Autoencoder
    autoencoder = keras.Model(
        inputs, decoder(encoder(inputs)), name='autoencoder')

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
