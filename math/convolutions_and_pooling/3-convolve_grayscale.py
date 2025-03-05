#!/usr/bin/env python3
"""striding a grayscale convolution"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
    kh is the height of the kernel
    kw is the width of the kernel
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
    if ‘same’, performs a same convolution
    if ‘valid’, performs a valid convolution
    if a tuple:
    ph is the padding for the height of the image
    pw is the padding for the width of the image
    the image should be padded with 0’s
    stride is a tuple of (sh, sw)
    sh is the stride for the height of the image
    sw is the stride for the width of the image
    Returns: a numpy.ndarray containing the convolved images"""
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    sh, sw = stride[0], stride[1]
    if padding == 'valid':
        ph, pw = 0, 0
    
    if type(padding) == tuple:
        ph, pw = padding[0], padding[1]
    
    img_pad = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    new_h = int((h + 2 * ph - kh) / sh) + 1
    new_w = int((w + 2 * pw - kw) / sw) + 1
    new_shape = (m, new_h, new_w)
    output = np.zeros(new_shape)

    for i in range(new_h):
        for j in range(new_w):
            output[:, i, j] = (
                kernel * img_pad[:,
                i * sh:((i * sh) + kh),
                j * sw:((j * sw) + kw)]).sum(axis=(1, 2))
    return output