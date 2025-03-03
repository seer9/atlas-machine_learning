#!/usr/bin/env python3
"""Optimize model"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    network: model to optimize
    alpha: learning rate
    beta1: first Adam optimization parameter
    beta2: second Adam optimization parameter
    """
    opt_adam = K.optimizers.Adam(
        alpha, beta1, beta2)

    network.compile(optimizer=opt_adam,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
