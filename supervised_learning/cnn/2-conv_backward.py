#!/usr/bin/env python3
"""performs back propagation over a convolutional layer of a nn"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """initialize dimensions of the previous layer"""
    m, h_new, w_new, c_new = dZ.shape

    """getting dimensions for the gradients"""
    m, h_prev, w_prev, c_prev = A_prev.shape

    """getting dimensions for the kernels"""
    kh, kw, c_prev, c_new = W.shape

    """initializing stride variables"""
    sh, sw = stride

    """padding values position"""
    pw, ph = padding[1], padding[0]

    """conditional for padding"""
    if padding == 'same':
        ph = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pw = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))
    if padding == 'valid':
        ph = 0
        pw = 0

    """initializing dA_prev, dW, db to the correct shapes"""
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    """pad A_prev and dA_prev"""
    A_prev_pad = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')
    dA_prev_pad = np.pad(
        dA_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    """loop over the batch of training examples"""
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        """loop over the height and width of the output volume"""
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    """find the corners of the current slice"""
                    vs = h * sh
                    ve = vs + kh
                    hs = w * sw
                    he = hs + kw

                    """use the corners to define the slice from a_prev_pad"""
                    a_slice = a_prev_pad[vs:ve, hs:he]
                    da_prev_pad[vs:ve, hs:he] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

        if padding == 'same':
            dA_prev[i, :, :, :] = da_prev_pad[ph:-ph, pw:-pw]
        if padding == 'valid':
            dA_prev[i, :, :, :] = da_prev_pad

    return dA_prev, dW, db
