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
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset()

    def tokenize_dataset(self):
        """Creates tokenizers using BertTokenizerFast"""

        # Load pre-trained BERT tokenizers for Portuguese and English
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased')

        return tokenizer_pt, tokenizer_en
