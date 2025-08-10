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
    for word in set(sentence):
        max_count = 0
        # For each reference, count the occurrences of the word
        for ref in references:
            count = ref.count(word)
            if count > max_count:
                max_count = count
        all_words[word] = max_count

    # Calculate the total number of matches
    total_matches = sum(all_words.values())

    # Calculate the precision
    if total_matches == 0:
        return 0.0
    precision = total_matches / sen_len

    # Calculate the brevity penalty
    ref_len = min(len(ref) for ref in references)
    if sen_len > ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = np.exp(1 - ref_len / sen_len)

    # Calculate the BLEU score
    bleu_score = brevity_penalty * precision
    return bleu_score
