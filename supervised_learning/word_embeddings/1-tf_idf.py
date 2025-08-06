#!/usr/bin/env python3
"""tf-idf embedding matrix"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """tf-idf embedding matrix

    Args:
        sentences: the sentences to embed
        vocab: the vocabulary to use

    Returns:
        np.ndarray: The tf-idf embedding matrix.
    """
    v = TfidfVectorizer(vocabulary=vocab)
    X = v.fit_transform(sentences)
    E = X.toarray()
    F = v.get_feature_names_out()
    return E, F
