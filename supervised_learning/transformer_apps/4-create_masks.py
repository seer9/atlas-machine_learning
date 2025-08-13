#!/usr/bin/env python3
"""create masks for the dataset"""
import tensorflow as tf


def create_masks(inputs, targets):
    """create masks for inputs and targets.

    Args:
        inputs: Tensor of shape (batch_size, seq_len_in)
        targets: Tensor of shape (batch_size, seq_len_out)
    Returns:
        encoder_mask: Mask for encoder inputs of shape:
        (batch_size, 1, 1, seq_len_in)
        combined_mask: Mask for decoder inputs of shape:
        (batch_size, 1, seq_len_out, seq_len_out)
        decoder_mask: Mask for decoder inputs of shape:
        (batch_size, 1, 1, seq_len_in)
    """
    # Encoder padding mask
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Look-ahead mask for decoder
    seq_len_out = targets.shape[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0)

    # Decoder target padding mask
    decoder_target_padding_mask = tf.cast(
        tf.math.equal(targets, 0), tf.float32)
    decoder_target_padding_mask = decoder_target_padding_mask[
        :, tf.newaxis, tf.newaxis, :]

    # Combined mask for the decoder's first attention block
    combined_mask = tf.maximum(look_ahead_mask, decoder_target_padding_mask)

    # Decoder padding mask for the second attention block
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    return encoder_mask, combined_mask, decoder_mask
