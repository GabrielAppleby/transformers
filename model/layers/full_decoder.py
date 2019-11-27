import tensorflow as tf

from model.layers.decoder_block import DecoderBlock
from model.utils.positional_encoding import positional_encoding


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderBlock(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, things):
        """
        Decodes the target sentence using the target generated so far as well as the enc_output.
        :param x: The target sentence generated so far.
        :param enc_output: The encoding output.
        :param training: boolean, for drop out.
        :param look_ahead_mask: Stops the model from looking ahead in x.
        :param padding_mask: Stops the model from attending to padding.
        :return: None.
        """
        x = things[0]
        enc_output = things[1]
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.dec_layers[i](
                (x, enc_output))


        # x.shape == (batch_size, target_seq_len, d_model)
        return x

    def compute_output_shape(self, input_shape):
        shp = input_shape
        shp[-1] = self.f_prime
        return shp