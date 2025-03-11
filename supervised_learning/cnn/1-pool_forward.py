#!/usr/bin/env python3
"""Pooling Forward Prop"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs forward propagation over a pooling layer of a neural network
    Args:
        A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev) input to the pooling
                layer
                - m: number of examples
                - h_prev: height of the previous layer
                - w_prev: width of the previous layer
                - c_prev: number of channels in the previous layer
        kernel_shape: tuple (kh, kw) size of the kernel for the pooling
                - kh: kernel height
                - kw: kernel width
        stride: tuple (sh, sw) containing the strides for the pooling
                - sh: stride for the height
                - sw: stride for the width
        mode: string ('max' or 'avg') indicates the type of pooling
    Returns: output of the pooling layer
    """
    """demensions of prev layer"""
    (m, h_prev, w_prev, c_prev) = A_prev.shape

    """demensions of kernels"""
    kh, kw = kernel_shape

    """stride"""
    sh, sw = stride

    """dimensions of the conv output volume"""
    h_out = int((h_prev - kh) / sh + 1)
    w_out = int((w_prev - kw) / sw + 1)

    """init output volume"""
    output = np.zeros((m, h_out, w_out, c_prev))

    """loop over x axis, y axis then channel"""
    for i in range(h_out):
        for j in range(w_out):
            if mode == 'max':
                output[:, i, j] = np.max(A_prev[:,
                                                i * sh:i * sh + kh,
                                                j * sw:j * sw + kw],
                                         axis=(1, 2))
            if mode == 'avg':
                output[:, i, j] = np.mean(A_prev[:,
                                                 i * sh:i * sh + kh,
                                                 j * sw:j * sw + kw],
                                          axis=(1, 2))
    return output
