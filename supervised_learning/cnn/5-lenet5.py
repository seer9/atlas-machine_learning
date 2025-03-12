#!/usr/bin/env python3
"""modified version of the LeNet-5 architecture with keras"""
from tensorflow import keras as K


def lenet5(X):
    """
    function that builds a version of the LeNet-5 architecture
    """
    conv1 = K.layers.Conv2D(
        filters=6, kernel_size=5,
        padding='same', activation='relu',
        kernel_initializer=K.initializers.HeNormal())(X)

    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = K.layers.Conv2D(
        filters=16, kernel_size=5,
        padding='valid', activation='relu',
        kernel_initializer=K.initializers.HeNormal())(pool1)

    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    flatten = K.layers.Flatten()(pool2)

    fc1 = K.layers.Dense(units=120, activation='relu',
                         kernel_initializer=K.initializers.HeNormal())(flatten)
    fc2 = K.layers.Dense(units=84, activation='relu',
                         kernel_initializer=K.initializers.HeNormal())(fc1)
    fc3 = K.layers.Dense(units=10,
                         kernel_initializer=K.initializers.HeNormal())(fc2)

    model = K.models.Model(X, fc3)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
