#!/usr/bin/env python3
"""calculating precision of classes in confusion matrix"""
import numpy as np


def precision(confusion):
    """determines precision with a ratio of TP over TP + FP"""
    classes = confusion.shape[0]
    precision = np.zeros(classes)

    for _ in range(classes):
        tp = confusion[_][_]
        p = np.sum(confusion[:, _])
        precision[_] = tp / p

    return precision
