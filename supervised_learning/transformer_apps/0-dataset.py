#!/usr/bin/env python3
"""dataset class"""
import transformers
import tensorflow_datasets as tfds


class Dataset():
    """class for loading and processing the dataset"""

    def __init__(self):
        """initializes the dataset"""

        # datasets
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)

        # tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset()

    def tokenize_dataset(self):
        """Creates tokenizers using BertTokenizerFast and trains them"""

        # load pre-trained BERT tokenizers 
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased')
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased')

        # helper function to decode and strip text
        def decode_and_strip(dataset, lang):
            for pt, en in dataset:
                if lang == 'pt':
                    text = pt.numpy().decode('utf-8')
                else:
                    text = en.numpy().decode('utf-8')
                yield text.strip() 

        # train the tokenizers
        tokenizer_pt.train_new_from_iterator(
            decode_and_strip(self.data_train, lang='pt'),
            vocab_size=2**13
        )
        tokenizer_en.train_new_from_iterator(
            decode_and_strip(self.data_train, lang='en'),
            vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
