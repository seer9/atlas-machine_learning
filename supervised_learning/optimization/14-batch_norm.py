#!/usr/bin/env python3
"""Batch Normalization"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for a neural network in tensorflow
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    
    Z = layer(prev)
    
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma')
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name='beta')
    
    epsilon = 1e-8
    
    mean, variance = tf.nn.moments(Z, axes=0)
    
    Z_norm = tf.nn.batch_normalization(
        x=Z,
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma, 
        variance_epsilon=epsilon)
    return activation(Z_norm)