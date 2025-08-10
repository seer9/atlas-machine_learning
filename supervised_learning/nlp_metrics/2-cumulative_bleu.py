#!/usr/bin/env python3
"""cumulative n-gram BLEU score"""
import numpy as np


def get_ngrams(sentence, n):
    """generate n-grams from a sentence"""
    return [tuple(sentence[i:i + n]) for i in range(len(sentence) - n + 1)]


def get_matches(references, sentence_ngrams, n):
    """count the matches of n-grams between sentence and references"""
    matches = {}
    for ngram in sentence_ngrams:
        max_count = 0
        for ref in references:
            ref_ngrams = get_ngrams(ref, n)
            max_count = max(max_count, ref_ngrams.count(ngram))
        if max_count > 0:
            matches[ngram] = max_count
    return matches


def cumulative_bleu(references, sentence, n):
    """calculate the cumulative n-gram BLEU score for a sentence against
    a list of reference sentences

    Args:
        references: list of reference sentences, each a list of words.
        sentence: the sentence to evaluate, a list of words.
        n: the maximum size of n-grams to use

    Returns: the cumulative n-gram BLEU score
    """
    sent_len = len(sentence)
    precisions = []

    for i in range(1, n + 1):
        sentence_ngrams = get_ngrams(sentence, i)
        matches = get_matches(references, sentence_ngrams, i)
        match_count = sum(matches.values())
        if sentence_ngrams:
            precision = match_count / len(sentence_ngrams)
        else:
            precision = 0.0
        precisions.append(precision)

    # calculate geometric mean
    if any(p == 0 for p in precisions):
        return 0.0
    geometric_mean = np.prod(precisions) ** (1 / n)

    # calulate brevity penalty
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(
        ref_lens, key=lambda ref_len: (abs(ref_len - sent_len), ref_len))
    brevity_penalty = np.exp(
        1 - closest_ref_len / sent_len) if sent_len < closest_ref_len else 1

    return geometric_mean * brevity_penalty
