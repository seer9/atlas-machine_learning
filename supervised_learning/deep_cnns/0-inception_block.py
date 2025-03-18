#!/usr/bin/env python3
"""function that builds an inception block"""
from tensorflow import keras as K

def inception_block(A_prev, filters):
    """
    Builds an inception block.

    Args:
        A_prev: The output from the previous layer.
        filters: A tuple or list containing F1, F3R, F3, F5R, F5, FPP, respectively:
            
    Returns:
        concatenated output of the inception block.
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 Convolution for local sparse structure approximation
    conv1x1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1), activation='relu')(A_prev)

    # 1x1 Convolution for dimension reduction before 3x3 Convolution
    conv3x3_reduce = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1), activation='relu')(A_prev)
    conv3x3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same', activation='relu')(conv3x3_reduce)

    # 1x1 Convolution for dimension reduction before 5x5 Convolution
    conv5x5_reduce = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1), activation='relu')(A_prev)
    conv5x5 = K.layers.Conv2D(filters=F5, kernel_size=(5, 5), padding='same', activation='relu')(conv5x5_reduce)

    # Max Pooling for multi-scale feature extraction, followed by 1x1 Convolution for dimension reduction
    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(A_prev)
    max_pool_conv = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1), activation='relu')(max_pool)

    # Concatenate all the outputs to aggregate multi-scale features
    output = K.layers.Concatenate()([conv1x1, conv3x3, conv5x5, max_pool_conv])

    return output
