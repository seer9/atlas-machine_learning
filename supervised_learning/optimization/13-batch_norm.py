#!/usr/bin/env python3
"""Batch Normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    normalizes an unactivated output of a neural network
    using batch norm
    """
    mean = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)
    """normalization from task 1"""
    Z_norm = (Z - mean) / np.sqrt(var + epsilon)
    """scale and shift"""
    Z_tilda = gamma * Z_norm + beta
    return Z_tilda
