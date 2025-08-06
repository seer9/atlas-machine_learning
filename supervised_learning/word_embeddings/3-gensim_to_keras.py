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
    return tf.keras.layers.Embedding(
        input_dim=model.wv.vectors.shape[0],
        output_dim=model.wv.vectors.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(model.wv.vectors),
        trainable=False
    )