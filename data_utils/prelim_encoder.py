import tensorflow_datasets as tfds


class PrelimEncoder:
    def __init__(self, train_examples, src_vocab_size=2**13, tgt_vocab_size=2**13):
        super(PrelimEncoder, self).__init__()
        self.tokenizer_src = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (src for src in train_examples[0]), target_vocab_size=src_vocab_size)

        self.tokenizer_tgt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (tgt for tgt in train_examples[1]), target_vocab_size=tgt_vocab_size)

    def encode(self, src, tgt):
        src = list(map(self.encode_src, src))
        tgt = list(map(self.encode_trg, tgt))

        return src, tgt

    def encode_src(self, src):
        return [self.tokenizer_src.vocab_size] + self.tokenizer_src.encode(
            src) + [self.tokenizer_src.vocab_size + 1]

    def encode_trg(self, tgt):
        return [self.tokenizer_tgt.vocab_size] + self.tokenizer_tgt.encode(
            tgt) + [self.tokenizer_tgt.vocab_size + 1]

    def get_src_vocab_size(self):
        return self.tokenizer_src.vocab_size + 2

    def get_tgt_vocab_size(self):
        return self.tokenizer_tgt.vocab_size + 2

    def get_src_start_token(self):
        return self.tokenizer_src.vocab_size

    def get_src_end_token(self):
        return self.tokenizer_src.vocab_size + 1

    def get_tgt_start_token(self):
        return self.tokenizer_tgt.vocab_size

    def get_tgt_end_token(self):
        return self.tokenizer_tgt.vocab_size + 1
