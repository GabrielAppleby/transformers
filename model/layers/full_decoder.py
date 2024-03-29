import tensorflow as tf

from model.layers.decoder_block import DecoderBlock
from model.utils.positional_encoding import positional_encoding
from data_utils.utils import create_look_ahead_mask, create_padding_mask


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, tgt_vocab_size,
                 maximum_position_encoding, rate=0.1, **kwargs):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.tgt_vocab_size = tgt_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        self.embedding = tf.keras.layers.Embedding(tgt_vocab_size, d_model, mask_zero=False)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderBlock(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, mask=None):
        tgt = inputs[0]
        tgt = tgt[:, :-1]
        enc_output = inputs[1]
        mask = mask[1]

        look_ahead_mask = create_look_ahead_mask(tf.shape(tgt)[1])
        dec_target_padding_mask = create_padding_mask(tgt)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        # tgt = tgt[:, :-1]
        seq_len = tf.shape(tgt)[1]
        x = self.embedding(tgt)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i]([x, enc_output], mask=[combined_mask, mask])

        # x.shape == (batch_size, target_seq_len, d_model)
        return x

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = super().get_config()
        config['num_layers'] = self.num_layers
        config['d_model'] = self.d_model
        config['num_heads'] = self.num_heads
        config['dff'] = self.dff
        config['tgt_vocab_size'] = self.tgt_vocab_size
        config['maximum_position_encoding'] = self.maximum_position_encoding
        config['rate'] = self.rate
        return config
