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
        x = keras.layers.Conv2D(f, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    latent = keras.layers.Conv2D(latent_dims[2], kernel_size=(3, 3), activation='relu', padding='same')(x)

    encoder = keras.Model(inputs, latent)

    x = latent
    for f in reversed(filters):
        x = keras.layers.Conv2DTranspose(f, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    outputs = keras.layers.Conv2D(input_dims[2], kernel_size=(3, 3), activation='sigmoid', padding='same')(x)

    decoder = keras.Model(latent, outputs, name='decoder')

    autoencoder = keras.Model(inputs, decoder(encoder(inputs)), name='autoencoder')

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
