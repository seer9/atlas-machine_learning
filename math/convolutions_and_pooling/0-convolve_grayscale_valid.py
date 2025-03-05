#!/usr/bin/env python3
"""performs a valid convolution on grayscale image"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """performs a valid convolution on grayscale image"""
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    nh = h - kh + 1
    nw = w - kw + 1
    output = np.zeros((m, nh, nw))
    for i in range(nh):
        for j in range(nw):
            image = images[:, i:(i+kh), j:(j+kw)]
            output[:, i, j] = np.sum(image * kernel, axis=(1, 2))
    return output
