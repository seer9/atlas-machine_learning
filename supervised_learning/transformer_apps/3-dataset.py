#!/usr/bin/env python3
"""dataset class"""
import transformers
import tensorflow_datasets as tfds
import tensorflow as tf


class Dataset():
    """class for loading and processing the dataset"""

    def __init__(self, batch_size, max_len):
        """initializes the dataset

        Args:
            batch_size (int): Batch size for training/validation
            max_len (int): Maximum number of tokens allowed per example sentence
        """

        def filter_max_length(pt, en):
            """Filter function to remove examples with sentences longer than max_len"""
            return tf.logical_and(tf.size(pt) <= max_len, tf.size(en) <= max_len)

        # Load datasets
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)

        # Load tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset()

        # Preprocess datasets
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        # Filter out examples with sentences longer than max_len
        self.data_train = self.data_train.filter(filter_max_length)
        self.data_valid = self.data_valid.filter(filter_max_length)

        # Cache, shuffle, batch, and prefetch the training dataset
        self.data_train = (self.data_train
                           .cache()
                           .shuffle(20000)
                           .padded_batch(batch_size, padded_shapes=([None], [None]))
                           .prefetch(tf.data.experimental.AUTOTUNE))

        # Batch the validation dataset
        self.data_valid = self.data_valid.padded_batch(
            batch_size, padded_shapes=([None], [None]))

    def tokenize_dataset(self):
        """Creates tokenizers using BertTokenizerFast and trains them"""

        try:
            # Load pre-trained BERT tokenizers
            tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
                'neuralmind/bert-base-portuguese-cased')
            tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
                'bert-base-uncased')

            # Train tokenizers on the dataset
            tokenizer_pt.train_new_from_iterator(
                (pt.numpy().decode('utf-8') for pt, _ in self.data_train),
                vocab_size=(2**13)
            )
            tokenizer_en.train_new_from_iterator(
                (en.numpy().decode('utf-8') for _, en in self.data_train),
                vocab_size=(2**13)
            )

            return tokenizer_pt, tokenizer_en

        except Exception as e:
            raise RuntimeError(f"Error loading tokenizers: {e}")

    def encode(self, pt, en):
        """Encodes Portuguese and English sentences into tokens."""
        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        # Decode tensors to strings
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        # Tokenize and add start/end tokens
        pt_tokens = [vocab_size_pt] + (
            self.tokenizer_pt.encode(pt_text) + [vocab_size_pt + 1])
        en_tokens = [vocab_size_en] + (
            self.tokenizer_en.encode(en_text) + [vocab_size_en + 1])

        # Convert to tensors
        return (tf.convert_to_tensor(pt_tokens, dtype=tf.int64),
                tf.convert_to_tensor(en_tokens, dtype=tf.int64))

    def tf_encode(self, pt, en):
        """Wrapper for the encode method."""
        pt_encoded, en_encoded = tf.py_function(
            self.encode, [pt, en], [tf.int64, tf.int64]
        )

        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])
        return tf.cast(pt_encoded, tf.int64), tf.cast(en_encoded, tf.int64)
