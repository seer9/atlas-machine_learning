#!/usr/bin/env python3
"""convolution on grayscale images with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    performs a convolution on grayscale images with custom padding.
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
        padding: (ph, pw) where
        - ph: padding for the height of the images
        - pw: padding for the width of the images
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    ph = padding[0]
    pw = padding[1]

    """getting convolution output dimensions"""
    nh = h - kh + 1 + (2 * ph)
    nw = w - kw + 1 + (2 * pw)
    output = np.zeros((m, nh, nw))
    img_pad = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    """loop over every pixel in the output"""
    for i in range(nh):
        """multiplying the kernel and the image pixels"""
        for j in range(nw):
            image = img_pad[:, i:(i+kh), j:(j+kw)]
            output[:, i, j] = np.sum(image * kernel, axis=(1, 2))
    return output
