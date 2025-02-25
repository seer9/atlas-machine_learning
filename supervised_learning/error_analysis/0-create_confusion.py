#!/usr/bin/env python3
"""Confusion Matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    creates a confusion matrix
    args:
    labels - the correct labels for each data point
    logits - the predicted labels
    """
    tlabels = np.argmax(labels, axis=1)
    plabels = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))

    for true, pred in zip(tlabels, plabels):
        confusion[true][pred] += 1

    return confusion
