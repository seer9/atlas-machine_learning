#!/usr/bin/env python3
"""n-gram BLEU score"""
import numpy as np


def get_ngrams(sentence, n):
    """Generate n-grams from a sentence."""
    return [tuple(sentence[i:i + n]) for i in range(len(sentence) - n + 1)]


def get_matches(references, sentence, n):
    """Count matches of n-grams in references, considering max counts."""
    sentence_ngrams = get_ngrams(sentence, n)
    ref_counts = {}

    for ref in references:
        ref_ngrams = get_ngrams(ref, n)
        for ngram in ref_ngrams:
            ref_counts[ngram] = max(
                ref_counts.get(ngram, 0), ref_ngrams.count(ngram))

    matches = {}
    for ngram in sentence_ngrams:
        if ngram in ref_counts:
            matches[ngram] = matches.get(ngram, 0) + 1
            matches[ngram] = min(matches[ngram], ref_counts[ngram])

    return matches


def ngram_bleu(references, sentence, n):
    """calculate the n-gram BLEU score for a sentence against a list of
    reference sentences.
    Args:
        references: list of reference sentences, each a list of words.
        sentence: the sentence to evaluate, a list of words.
        n: the size of n-grams to use.
    Returns: the n-gram BLEU score.
    """
    sen_len = len(sentence)
    if sen_len == 0:
        return 0.0

    matches = get_matches(references, sentence, n)

    # Calculate the total number of matches
    total_matches = sum(matches.values())

    # Calculate the precision
    if (sen_len - n + 1) > 0:
        precision = total_matches / (sen_len - n + 1)
    else:
        precision = 0.0

    # Calculate the brevity penalty
    ref_len = min(len(ref) for ref in references)
    if sen_len > ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = np.exp(1 - ref_len / sen_len)

    # Calculate the BLEU score
    bleu_score = brevity_penalty * precision
    return bleu_score
