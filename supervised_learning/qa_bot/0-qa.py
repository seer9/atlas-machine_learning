#!/usr/bin/env python3
"""Question answering bot Project"""
import tensorflow as tf
from transformers import BertTokenizer
import tensorflow_hub as hub


def question_answer(question, reference):
    """Finds a snippet of text within a reference document to answer a question.
    Args:
        question: The question to answer.
        reference: The reference document to search.
    Returns:
        A string representing the answer to the question or None if not found.
    """
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

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
    answer_tokens = input_tensors[0][0, start_index:end_index + 1]
    answer = tokenizer.decode(
        answer_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    # No answer ? Returning None
    if not answer.strip():
        return None

    return answer