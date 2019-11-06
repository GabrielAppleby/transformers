import tensorflow as tf

from model.layers.full_decoder import Decoder
from model.layers.full_encoder import Encoder


class Transformer(tf.keras.Model):
    def __init__(self,
                 input_vocab_size,
                 target_vocab_size,
                 pe_input,
                 pe_target,
                 num_blocks,
                 d_model,
                 num_heads,
                 dff,
                 drop_rate):
        """
        Initializes the transformer model.
        :param input_vocab_size: The size of your input vocab.
        :param target_vocab_size: The size of your target vocab size.
        :param pe_input: The maximum positional encoding of the input. I'm not sure when this should
        differ from the input_vocab_size.
        :param pe_target: The maximum positional encoding of the output. I'm not sure when this
        should differ from the target_vocab_size.
        :param num_blocks: The number of encoder blocks to use within the encoder. Default is 6.
        :param d_model: The size of the "embedding" (not sure if I can call that an embedding) that
        is passed around. Default is 512.
        :param num_heads: The number of heads to use within the multi-head attention. Default is 8.
        :param dff: The number of units for the first layer of feed forward networks. Default is
        2048.
        :param drop_rate: The dropout rate for all drop out used.
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            num_blocks, d_model, num_heads, dff, input_vocab_size, pe_input, drop_rate)

        self.decoder = Decoder(num_blocks, d_model, num_heads, dff,
                               target_vocab_size, pe_target, drop_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self,
             inp,
             tar,
             training,
             enc_padding_mask,
             look_ahead_mask,
             dec_padding_mask):
        """
        Call the transformer
        :param inp: The input text.
        :param tar: The target text.
        :param training: The boolean indicating whether we are training or not.
        :param enc_padding_mask: Really not sure.
        :param look_ahead_mask: Really not sure.
        :param dec_padding_mask: Really not sure.
        :return: Nothing.
        """
        # (batch_size, inp_seq_len, depth)
        enc_output = self.encoder(inp, training, enc_padding_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights
