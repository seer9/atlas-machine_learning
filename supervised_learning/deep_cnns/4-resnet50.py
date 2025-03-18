#!/usr/bin/env python3
"""builds the ResNet-50 architecture"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """builds the ResNet-50 architecture"""
    X = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(
        64, kernel_size=(7, 7), strides=(2, 2),
        padding='same', kernel_initializer='he_normal')(X)
    batch1 = K.layers.BatchNormalization()(conv1)
    act1 = K.layers.Activation('relu')(batch1)
    pool1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(act1)
    filters = [64, 64, 256]
    Y = projection_block(pool1, filters, s=1)

    for i in range(2):
        Y = identity_block(Y, filters)
    filters = [128, 128, 512]
    Y = projection_block(Y, filters)

    for i in range(3):
        Y = identity_block(Y, filters)
    filters = [256, 256, 1024]
    Y = projection_block(Y, filters)

    for i in range(5):
        Y = identity_block(Y, filters)
    filters = [512, 512, 2048]
    Y = projection_block(Y, filters)

    for i in range(2):
        Y = identity_block(Y, filters)
    avg_pool = K.layers.AveragePooling2D((7, 7), padding='same')(Y)
    Y = K.layers.Dense(1000, activation='softmax')(avg_pool)

    model = K.models.Model(inputs=X, outputs=Y)
    return model
