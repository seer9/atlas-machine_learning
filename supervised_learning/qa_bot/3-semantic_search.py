#!/usr/bin/env python3
"""semantic search function"""
from sentence_transformers import SentenceTransformer
import numpy as np
import os


def semantic_search(corpus_path, sentence):
    """Perform semantic search on a corpus of documents.

    Args:
        corpus_path (str): Path to the directory containing text files.
        sentence (str): The query sentence to search for.

    Returns:
        str: The most relevant document or None if no documents are found.
    """
    # load trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # read all docs in corpus
    docs = []
    for filename in os.listdir(corpus_path):
        if filename.endswith('.md'):
            files_path = os.path.join(corpus_path, filename)
            with open(files_path, 'r') as file:
                docs.append(file.read())

    # generate embeddings for the documents
    doc_embeddings = model.encode(docs)

    # generate embedding for the input sentence
    sentence_embedding = model.encode([sentence])[0]

    # calculate cosine similarities
    similarities = np.dot(doc_embeddings, sentence_embedding.T).flatten()

    # Return most similar document
    most_similar_idx = np.argmax(similarities)
    return docs[most_similar_idx]