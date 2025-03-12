#!/usr/bin/env python3
"""function that performs back propagation over a pooling layer of a nn"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs back propagation over a pooling layer of a nn"""
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vs = h * sh
                    ve = vs + kh
                    hs = w * sw
                    he = hs + kw

                    if mode == 'max':
                        a_slice = a_prev[vs:ve, hs:he, c]
                        mask = (a_slice == np.max(a_slice))
                        dA_prev[i, vs:ve, hs:he, c] += np.multiply(
                            mask, dA[i, h, w, c])
                    if mode == 'avg':
                        da = dA[i, h, w, c]
                        avg = da / (kh * kw)
                        Z = np.ones(kernel_shape) * avg
                        dA_prev[i, vs:ve, hs:he, c] += Z
    return dA_prev
