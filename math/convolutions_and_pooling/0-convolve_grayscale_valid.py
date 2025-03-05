#!/usr/bin/env python3
"""valid convolution on grayscale image"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    performs a valid convolution on grayscale image.
    Args:
        images: multiple grayscale images
        shape (m, h, w) where
        - m: number of images
        - h: height in pixels of the images
        - w: width in pixels of the images
        kernel: kernel for the convolution
        shape (kh, kw) where
        - kh: height of the kernel
        - kw: width of the kernel
    """
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
