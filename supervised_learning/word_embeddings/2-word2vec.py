#!/usr/bin/env python3
"""word2vec model"""
import gensim

def word2vec_model(sentences, vector_size=100, min_count=5, window=5, negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create a word2vec model

    Args:
        sentences: the sentences to train on
        vector_size: the dimensionality of the word vectors
        min_count: ignores all words with total frequency lower than this.
        window: the maximum distance between the current and predicted word within a sentence.
        negative: the int for negative specifies how many "noise words"
        cbow: determines the training type. If True, CBOW; else, Skip-gram
        epochs: number of iterations over the corpus
        seed: random seed for reproducibility
        workers: number of worker threads to train the model

    Returns:
        Word2Vec: The trained Word2Vec model.
    """
    if cbow is True:
        cbow_flag = 0
    else:
        cbow_flag = 1

    model = gensim.models.Word2Vec(sentences=sentences,
                     vector_size=vector_size,
                     min_count=min_count,
                     window=window,
                     negative=negative,
                     sg=cbow_flag,
                     epochs=epochs,
                     seed=seed,
                     workers=workers)
    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    return model