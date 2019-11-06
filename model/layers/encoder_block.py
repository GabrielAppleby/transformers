import tensorflow as tf

from model.inner_models import point_wise_feed_forward_network
from model.layers.multi_head_attention import MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        Initializes the EncoderBlock.
        :param d_model: The size of the embedding.
        :param num_heads: The number of heads of attention.
        :param dff: The number of
        :param rate:
        """
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """
        Performs multihead attention on the input, then drop out, then layer norm, and then sends
        it through a feedforward, before dropout again followed by another layer norm.
        :param x: The input.
        :param training: boolean, for drop out.
        :param mask: The padding mask.
        :return: Nothing.
        """
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2
