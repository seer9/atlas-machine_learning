#!/usr/bin/env python3
"""calculating sensitivity for classes in a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """determines sensitivity with a ratio of TP over P"""
    classes = confusion.shape[0]
    sensitivity = np.zeros(classes)

    for _ in range(classes):
        tp = confusion[_][_]
        p = np.sum(confusion[_])
        sensitivity[_] = tp / p

    return sensitivity
