import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds


class PrelimEncoder:
    def __init__(self, train_examples, inpt_vocab_size=2 ** 13, trgt_vocab_size=2 ** 13):
        super(PrelimEncoder, self).__init__()
        self.tokenizer_inpt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (inpt for inpt in train_examples[0]), target_vocab_size=inpt_vocab_size)

        self.tokenizer_trgt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (trgt for trgt in train_examples[1]), target_vocab_size=trgt_vocab_size)

    def encode(self, langs):

        lang1 = langs[0]
        lang2 = langs[1]

        lang1 = list(map(self.encode_inpt_two, lang1))
        lang2 = list(map(self.encode_trgt_two, lang2))

        return lang1, lang2

    def encode_inpt_two(self, inpt_lang):
        return [self.tokenizer_inpt.vocab_size] + self.tokenizer_inpt.encode(
            inpt_lang) + [self.tokenizer_inpt.vocab_size + 1]

    def encode_trgt_two(self, trgt_lang):
        return [self.tokenizer_trgt.vocab_size] + self.tokenizer_trgt.encode(
            trgt_lang) + [self.tokenizer_trgt.vocab_size + 1]

    def encode_inpt(self, inpt_lang):
        return self.tokenizer_inpt.encode(inpt_lang)

    def encode_trgt(self, trgt_lang):
        return self.tokenizer_trgt.encode(trgt_lang)

    def decode_inpt(self, inpt_vec):
        return self.tokenizer_inpt.decode(inpt_vec)

    def decode_trgt(self, trgt_vec):
        return self.tokenizer_trgt.decode(trgt_vec)

    def get_inpt_vocab_size(self):
        return self.tokenizer_inpt.vocab_size + 2

    def get_trgt_vocab_size(self):
        return self.tokenizer_trgt.vocab_size + 2

    def get_inpt_start_token(self):
        return self.tokenizer_inpt.vocab_size

    def get_inpt_end_token(self):
        return self.tokenizer_inpt.vocab_size + 1

    def get_trgt_start_token(self):
        return self.tokenizer_trgt.vocab_size

    def get_trgt_end_token(self):
        return self.tokenizer_trgt.vocab_size + 1
