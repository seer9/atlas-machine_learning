#!/usr/bin/env python3
"""using striding with grayscale convolution"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    using striding with grayscale convolution.
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
        padding: 'same' or 'valid'
        stride: tuple (sh, sw) where
        - sh: stride for the height of the images
        - sw: stride for the width of the images
    """
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    sh, sw = stride[0], stride[1]

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1

    if padding == 'valid':
        ph = 0
        pw = 0

    img_pad = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    new_h = int((h + 2 * ph - kh) / sh) + 1
    new_w = int((w + 2 * pw - kw) / sw) + 1
    new_shape = (m, new_h, new_w)
    output = np.zeros(new_shape)

    for i in range(new_h):
        for j in range(new_w):
            output[:, i, j] = (
                kernel * img_pad[
                    :,
                    i * sh:((i * sh) + kh),
                    j * sw:((j * sw) + kw)]).sum(axis=(1, 2))
    return output
