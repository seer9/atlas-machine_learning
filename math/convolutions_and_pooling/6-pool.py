#!/usr/bin/env python3
"""performs pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """performs pooling on images.
    Args:
        images: containing multiple images
                - m: number of images
                - h: height in pixels of the images
                - w: width in pixels of the images
                - c: number of channels in the image
        kernel_shape: containing the kernel shape for the pooling
                - kh: height of the kernel
                - kw: width of the kernel
        stride: tuple of (sh, sw) containing the strides for the pooling
                - sh: stride for the height
                - sw: stride for the width
        mode: indicates the type of pooling
                - max: indicates max pooling
                - avg: indicates average pooling
    Returns: the pooled images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    sh = stride[0]
    sw = stride[1]

    new_h = int((h - kh) / sh + 1)
    new_w = int((w - kw) / sw + 1)
    output = np.zeros((m, new_h, new_w, c))

    for i in range(new_h):
        for j in range(new_w):
            if mode == 'max':
                output[:, i, j] = np.max(images[:,
                                                i * sh:i * sh + kh,
                                                j * sw:j * sw + kw],
                                         axis=(1, 2))
            if mode == 'avg':
                output[:, i, j] = np.mean(images[:,
                                                 i * sh:i * sh + kh,
                                                 j * sw:j * sw + kw],
                                          axis=(1, 2))
    return output
