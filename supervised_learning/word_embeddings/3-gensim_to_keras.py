#!/usr/bin/env python3
"""Convert gensim word2vec model to Keras Embedding layer"""
import tensorflow as tf


def gensim_to_keras(model):
    """Convert gensim word2vec model to Keras Embedding layer

    Args:
        model: the gensim word2vec model

    Returns:
        keras.layers.Embedding: The Keras Embedding layer.
    """
    kv = model.wv
    w = kv.vectors
    layers = tf.keras.layers.Embedding(
        input_dim=kv.vectors.shape[0],
        output_dim=kv.vectors.shape[1],
        weights=[w],
        trainable=True,
    )
    return layers
