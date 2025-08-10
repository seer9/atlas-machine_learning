#!/usr/bin/env python3
"""cumulative n-gram BLEU score"""
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


def cumulative_bleu(references, sentence, n):
    """Calculate the cumulative n-gram BLEU score for a sentence against a list of
    reference sentences.

    Args:
        references: list of reference sentences, each a list of words.
        sentence: the sentence to evaluate, a list of words.
        n: the maximum size of n-grams to use.

    Returns: the cumulative n-gram BLEU score.
    """
    sen_len = len(sentence)
    if sen_len == 0:
        return 0.0

    # Calculate precision for each n-gram size 
    precisions = []
    for k in range(1, n + 1):
        matches = get_matches(references, sentence, k)

        # Calculate the total number of matches
        total_matches = sum(matches.values())

        # Calculate the precision for k-grams
        possible_ngrams = max(1, sen_len - k + 1) 
        precision = total_matches / possible_ngrams
        precisions.append(precision)

    # Calculate the geometric mean of precisions
    if all(p == 0 for p in precisions):
        geometric_mean = 0.0
    else:
        geometric_mean = np.exp(np.sum(np.log(p) for p in precisions if p > 0) / n)

    # Calculate the brevity penalty
    ref_lens = [len(ref) for ref in references]
    ref_len = min(ref_lens, key=lambda r: (abs(r - sen_len), r))
    if sen_len > ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = np.exp(1 - ref_len / sen_len)

    # Calculate the BLEU score
    bleu_score = brevity_penalty * geometric_mean
    return bleu_score