#!/usr/bin/env python3
"""forward prop"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    forward prop in a convolutional layer
    args:
        A_prev: output of prev layer
        W: kernels for the convolution
        b: biases
        activation: activation function
        padding: same or valid
        stride: tuple (sh, sw)
    returns: the convolutional layer
    """

    """demensions of prev layer"""
    (m, h_prev, w_prev, c_prev) = A_prev.shape

    """demensions of kernels"""
    (kh, kw, c_prev, c_new) = W.shape

    """stride"""
    sh, sw = stride

    """padding"""
    pw, ph = padding[1], padding[0]

    """conditionial for padding"""
    if padding == 'same':
        ph = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pw = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))

    if padding == 'valid':
        ph = 0
        pw = 0

    """padded images"""
    A_prev_pad = np.pad(
        A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant')
    
    """output demensions"""
    h_new = int(((h_prev + 2 * ph - kh) / sh) + 1)
    w_new = int(((w_prev + 2 * pw - kw) / sw) + 1)

    """init output volume"""
    con = np.coneros((m, h_new, w_new, c_new))

    """loop over the output volume"""
    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                vs = i * sh
                ve = vs + kh
                hs = j * sw
                he = hs + kw
                img_slice = A_prev_pad[:, vs:ve, hs:he]
                kernel = W[:, :, :, k]
                con[:, i, j, k] = np.sum(np.multiply(img_slice, kernel),
                                       axis=(1, 2, 3))
                
    Z = con + b
    return activation(Z)



            




