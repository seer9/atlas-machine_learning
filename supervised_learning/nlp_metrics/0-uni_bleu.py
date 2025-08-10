#!/usr/bin/env python3
"""BLEU score calculation"""
import numpy as np


def uni_bleu(references, sentence):
    """calculate the unigram BLEU score for a sentence against a list of
        reference sentences.

    Args:
        references: list of reference sentences, each a list of words.
        sentence: the sentence to evaluate, a list of words.

    Returns: the unigram BLEU score.
    """
    sen_len = len(sentence)
    ref_len = []
    all_words = {}

    # Count occurrences of each word in the sentence
    for word in sentence:
        if word in all_words:
            all_words[word] += 1
        else:
            all_words[word] = 1

    # Count occurrences of each word in the references
    for ref in references:
        ref_len.append(len(ref))
        for word in ref:
            if word in all_words:
                all_words[word] += 1
            else:
                all_words[word] = 1

    # Calculate the number of matches
    matches = 0
    for word, count in all_words.items():
        if count > 1:
            matches += count - 1
    # Calculate the brevity penalty
    ref_len = np.array(ref_len)
    min_ref_len = ref_len.min()
    if sen_len > min_ref_len:
        bp = 1
    else:
        bp = np.exp(1 - (min_ref_len / sen_len))
    # Calculate the BLEU score
    if sen_len == 0:
        return 0.0
    bleu_score = bp * (matches / sen_len)
    return bleu_score
