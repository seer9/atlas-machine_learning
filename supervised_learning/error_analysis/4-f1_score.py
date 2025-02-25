#!/usr/bin/env python3
"""F1 score of a confusion matrix"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """determines the F1 score with a ratio of 2 * TP over 2 * TP + FP + FN"""
    classes = confusion.shape[0]
    f1 = np.zeros(classes)

    ppv = precision(confusion)
    tpr = sensitivity(confusion)

    for i in range(classes):
        f1[i] = 2 * ppv[i] * tpr[i] / (ppv[i] + tpr[i])

    return f1
