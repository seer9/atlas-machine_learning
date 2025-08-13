#!/usr/bin/env python3
"""dataset class"""
import transformers
import tensorflow_datasets as tfds


class Dataset():
    """class for loading and processing the dataset"""

    def __init__(self):
        """initializes the dataset"""

        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for the dataset"""

        tokenizer_pt = transformers.BertTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.BertTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        for pt, en in data:
            tokenizer_pt.tokenize(pt.numpy().decode('utf-8'))
            tokenizer_en.tokenize(en.numpy().decode('utf-8'))

        return tokenizer_pt, tokenizer_en
