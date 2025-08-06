#!/usr/bin/env python3
"""Bag of words embedding matrix"""
import numpy as np


def bag_of_words(sentence, vocab=None):
    """Bag of words embedding matrix

    Args:
        sentence (str): The sentence to embed.
        vocab (list, optional): The vocabulary to use. Defaults to None.

    Returns:
        np.ndarray: The bag of words embedding matrix.
    """
    if vocab is None:
        vocab = sorted(set(word for sentence in sentence for word in sentence.split()))
    else:
        vocab = sorted(vocab)

    features = vocab
    embeddings = np.zeros((len(sentence), len(vocab)), dtype=int)

    for i, sent in enumerate(sentence):
        words = sent.split()
        for word in words:
            if word in vocab:
                embeddings[i, vocab.index(word)] += 1

    return embeddings, features