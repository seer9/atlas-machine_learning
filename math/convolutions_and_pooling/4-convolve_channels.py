#!/usr/bin/env python3
"""convolution on images with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """performs a convolution on images with channels
    Args:
        images: shape containing multiple images
                m: number of images
                h: height in pixels
                w: width in pixels
                c: number of channels 
        kernel: contains the kernel/filter for the convolution
                kh: height of the kernel
                kw: width of the kernel
        padding: string that indicates the type of padding
                same: same padding
                valid: valid padding
        stride: tuple of (sh, sw) containing the strides for the convolution
                sh: stride for the height
                sw: stride for the width
        Returns: numpy.ndarray containing the convolved images"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh, kw = kernel.shape[0], kernel.shape[1]
    sh, sw = stride[0], stride[1]
    ph, pw = 0, 0

    """convultion types"""
    if padding == 'same':
        ph = int((((h - 1) * sh + kh - h) / 2) + 1)
        pw = int((((w - 1) * sw + kw - w) / 2) + 1)

    if padding == 'valid':
        ph = 0
        pw = 0

    img_padded = np.pad(
        images, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')

    ch = int(((h + 2 * ph - kh) / sh) + 1)
    cw = int(((w + 2 * pw - kw) / sw) + 1)

    """convolution of the images"""
    conv = np.zeros((m, ch, cw))

    for i in range(ch):
        for j in range(cw):
            images_slide = img_padded[
                :, 
                i * sh:i * sh + kh, 
                j * sw:j * sw + kw, :]
            conv[:, i, j] = np.sum(images_slide * kernel, axis=(1, 2, 3))
    return conv
