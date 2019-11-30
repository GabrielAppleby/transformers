import tensorflow as tf

from data_utils.utils import create_padding_mask
from model.layers.encoder_block import EncoderBlock
from model.utils.positional_encoding import positional_encoding


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, src_vocab_size,
                 maximum_position_encoding, rate=0.1, **kwargs):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.src_vocab_size = src_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        self.embedding = tf.keras.layers.Embedding(src_vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderBlock(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, mask=None):
        seq_len = tf.shape(x)[1]
        mask = create_padding_mask(x)

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask=mask)

        # (batch_size, input_seq_len, d_model)
        return x

    def compute_mask(self, inputs, mask=None):
        return create_padding_mask(inputs)

    def get_config(self):
        config = super().get_config()
        config['num_layers'] = self.num_layers
        config['d_model'] = self.d_model
        config['num_heads'] = self.num_heads
        config['dff'] = self.dff
        config['src_vocab_size'] = self.src_vocab_size
        config['maximum_position_encoding'] = self.maximum_position_encoding
        config['rate'] = self.rate
        return config
