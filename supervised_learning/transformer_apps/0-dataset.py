#!/usr/bin/env python3
"""dataset class"""
import transformers
import tensorflow_datasets as tfds


class Dataset():
    """class for loading and processing the dataset"""

    def __init__(self):
        """initializes the dataset"""

        # Load datasets
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)

        # Create tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for the dataset"""

        # Use SubwordTextEncoder to build tokenizers
        SubwordTextEncoder = tfds.deprecated.text.SubwordTextEncoder

        # Build Portuguese tokenizer
        tokenizer_pt = SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data),
            target_vocab_size=(2**13)
        )

        # Build English tokenizer
        tokenizer_en = SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data),
            target_vocab_size=(2**13)
        )

        return tokenizer_pt, tokenizer_en
