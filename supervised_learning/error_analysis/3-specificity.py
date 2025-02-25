#!/usr/bin/env python3
"""specificity of classes in a confusion matrix"""
import numpy as np


def specificity(confusion):
    """determines specificity with a ratio of TN over N"""
    classes = confusion.shape[0]
    specificity = np.zeros(classes)

    for _ in range(classes):
        tp = confusion[_][_]
        fp = np.sum(confusion[:, _]) - tp
        fn = np.sum(confusion[_]) - tp
        tn = np.sum(confusion) - tp - fp - fn
        specificity[_] = tn / (tn + fp)

    return specificity
