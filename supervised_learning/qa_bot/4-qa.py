#!/usr/bin/env python3
"""Question answering bot Project"""
import tensorflow as tf
from transformers import BertTokenizer
import tensorflow_hub as hub
import os
import numpy as np
from sentence_transformers import SentenceTransformer

tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
stf = SentenceTransformer('all-MiniLM-L6-v2')


def semantic_search(corpus_path, sentence):
    """Perform semantic search on a corpus of documents.

    Args:
        corpus_path (str): Path to the directory containing text files.
        sentence (str): The query sentence to search for.

    Returns:
        str: The most relevant document or None if no documents are found.
    """
    # read all docs in corpus
    docs = []
    for filename in os.listdir(corpus_path):
        if filename.endswith('.md'):
            files_path = os.path.join(corpus_path, filename)
            with open(files_path, 'r') as file:
                docs.append(file.read())

    # generate embeddings for the documents
    doc_embeddings = stf.encode(docs)

    # generate embedding for the input sentence
    sentence_embedding = stf.encode([sentence])[0]

    # calculate cosine similarities
    similarities = np.dot(doc_embeddings, sentence_embedding.T).flatten()

    # Return most similar document
    most_similar_idx = np.argmax(similarities)
    return docs[most_similar_idx]


def gets_answer(question, reference):
    """Finds a snippet of text within a reference document to answer a question
    Args:
        question: The question to answer.
        reference: The reference document to search.
    Returns:
        A string representing the answer to the question or None if not found.
    """
    # Tokenize the input
    inputs = tokenizer(question, reference, return_tensors="tf")

    # Extract input tensors
    input_tensors = [
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["token_type_ids"]
    ]

    # Pass the input tensors to the model
    output = model(input_tensors)

    # initialize the start and end
    start_logits = output[0]
    end_logits = output[1]

    # Determine the best start and end indices
    sequence_length = tf.shape(input_tensors[0])[1]
    start_index = tf.math.argmax(start_logits[0, 1:sequence_length - 1]) + 1
    end_index = tf.math.argmax(end_logits[0, 1:sequence_length - 1]) + 1

    # Extract and decode the answer tokens
    a_tokens = input_tensors[0][0, start_index:end_index + 1]
    answer = tokenizer.decode(
        a_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # No answer, Returns None
    if not answer.strip():
        return None

    return answer


def question_answer(coprus_path):
    """Answers questions from referred to text"""
    kill = ["exit", "quit", "bye", "goodbye"]
    input_text = ""

    while True:
        input_text = input("Q: ")

        if input_text.lower() in kill:
            print("A: Goodbye\n")
            exit()

        bdoc = semantic_search(coprus_path, input_text)

        answer = gets_answer(input_text, bdoc)
        if answer is None:
            print("A: I don't know the answer to that question.\n")
        else:
            print(f"A: {answer}\n")
