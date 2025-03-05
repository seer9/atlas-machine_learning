#!/usr/bin/env python3
"""same convolution on grayscale image"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    performs a same convolution on grayscale image.
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

    """adding padding to the image"""
    ph = int((kh - 1) / 2)
    pw = int((kw - 1) / 2)

    if kh % 2 == 0:
        ph = int(kh / 2)
    if kw % 2 == 0:
        pw = int(kw / 2)

    images = np.pad(
        images, pad_width=((0, 0), (ph, ph), (pw, pw)), mode='constant')
    """getting convolution output dimensions"""
    nh = h
    nw = w
    output = np.zeros((m, nh, nw))

    for i in range(nh):
        for j in range(nw):
            image = images[:, i:(i+kh), j:(j+kw)]
            output[:, i, j] = np.sum(image * kernel, axis=(1, 2))
    return output