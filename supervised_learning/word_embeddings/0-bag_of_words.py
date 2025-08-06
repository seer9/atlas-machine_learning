#!/usr/bin/env python3
"""bag of words embedding matrix"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentence, vocab=None):
    """bag of words embedding matrix

    Args:
        sentence: the sentence to embed
        vocab: the vocabulary to use

    Returns:
        np.ndarray: The bag of words embedding matrix.
    """
    v = CountVectorizer(vocabulary=vocab)
    X = v.fit_transform(sentence)
    E = X.toarray()
    F = v.get_feature_names_out()
    return E, F
