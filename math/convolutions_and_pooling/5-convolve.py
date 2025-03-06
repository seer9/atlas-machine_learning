#!/usr/bin/env python3
"""performs a convolution on images using multiple kernels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """performs a convolution on images using multiple kernels
    Args:
        images: containing multiple grayscale images
                - m: number of images
                - h: height in pixels of the images
                - w: width in pixels of the images
                - c: number of channels in the image
        kernel: containing the kernels for the convolution
                - kh: height of the kernel
                - kw: width of the kernel
                - nc: number of kernels
        padding: a tuple of (ph, pw), 'same' or 'valid'
                - if 'same', performs a same convolution
                - if 'valid', performs a valid convolution
                - where
                    - ph is the padding for the height of the image
                    - pw is the padding for the width of the image
                    - the image should be padded with 0’s
        stride: tuple of (sh, sw) containing the strides for the convolution
                - sh: stride for the height
                - sw: stride for the width
    Returns: the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernels.shape[0]
    kw = kernels.shape[1]
    nc = kernels.shape[3]
    sh, sw = stride[0], stride[1]
    pw, ph = padding[1], padding[0]

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1

    if padding == 'valid':
        ph = 0
        pw = 0

    img_pad = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant')

    new_h = int((h + 2 * ph - kh) / sh) + 1
    new_w = int((w + 2 * pw - kw) / sw) + 1
    new_shape = (m, new_h, new_w, nc)
    output = np.zeros(new_shape)
    image = np.arange(m)

    for i in range(new_h):
        for j in range(new_w):
            for k in range(nc):
                output[image, i, j, k] = (
                    kernels[:, :, :, k] * img_pad[
                        :,
                        i * sh:((i * sh) + kh),
                        j * sw:((j * sw) + kw), :]).sum(axis=(1, 2, 3))
    return output
