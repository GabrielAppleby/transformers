import tensorflow as tf

from model.layers.full_decoder import Decoder
from model.layers.encoder_block import EncoderBlock
from model.layers.full_encoder import Encoder
from model.layers.multi_head_attention import MultiHeadAttention
from model.utils.attention import scaled_dot_product_attention


class DropFirstColumn(tf.keras.layers.Layer):
    def __init__(self):
        super(DropFirstColumn, self).__init__()

    def call(self, x):
        return x[:, :-1]


class AttLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, batch_size):
        self.num_heads = num_heads
        assert d_model % self.num_heads == 0

        self.d_model = d_model
        self.batch_size = batch_size
        self.depth = d_model // self.num_heads
        super(AttLayer, self).__init__()

    def call(self, inpts):
        q = inpts[0]
        k = inpts[1]
        v = inpts[2]
        self.blah = q.shape[1]
        q = self.split_heads(q)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention = scaled_dot_product_attention(q, k, v)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention_enc = tf.reshape(scaled_attention, (self.batch_size, self.blah, self.d_model))

        return concat_attention_enc

    def compute_output_shape(self, input_shape):
        return (self.batch_size, self.blah, self.depth)

    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        :param x: The actual input.
        :param batch_size: The batch size.
        :return: None.
        """
        x = tf.reshape(x, (self.batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])


def get_transformer(input_vocab_size, target_vocab_size, pe_input, pe_target, num_blocks, d_model, num_heads, dff, drop_rate, pad_shape):
    assert d_model % num_heads == 0
    batch_size = 64
    depth = d_model // num_heads

    src = tf.keras.layers.Input(batch_size=64, shape=[pad_shape])
    tgt = tf.keras.layers.Input(batch_size=64, shape=[pad_shape])
    # tgt_real = DropFirstColumn()(tgt)

    src_embed = tf.keras.layers.Embedding(input_vocab_size, d_model)(src)
    enc_out2 = src_embed
    # adding embedding and position encoding.
    # (batch_size, inp_seq_len, depth)

    enc_out2 = Encoder(num_blocks, d_model, num_heads, dff, input_vocab_size, 10)(enc_out2)

    # for _ in range(num_blocks):
    #     q_enc = tf.keras.layers.Dense(d_model)(enc_out2)
    #     k_enc = tf.keras.layers.Dense(d_model)(enc_out2)
    #     v_enc = tf.keras.layers.Dense(d_model)(enc_out2)
    #
    #     # (batch_size, seq_len_q, d_model)
    #     concat_attention_enc = AttLayer(num_heads, d_model, batch_size)([q_enc, k_enc, v_enc])
    #
    #     attn_out_enc = tf.keras.layers.Dense(d_model)(concat_attention_enc)  # (batch_size, seq_len_q, d_model)
    #
    #
    #     attn_out_enc = tf.keras.layers.Dropout(.01)(attn_out_enc)
    #     enc_out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(enc_out2 + attn_out_enc)
    #     ffn_output_enc = tf.keras.layers.Dense(dff, activation='relu')(enc_out1)
    #     ffn_output_enc = tf.keras.layers.Dense(d_model)(ffn_output_enc)
    #     ffn_output_enc = tf.keras.layers.Dropout(.01)(ffn_output_enc)
    #     enc_out2 = ffn_output_enc
    #     enc_out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(enc_out1 + ffn_output_enc)

    tgt_embed = tf.keras.layers.Embedding(target_vocab_size, d_model)(tgt)
    dec_out3 = tgt_embed
    for _ in range(num_blocks):
        q_dec_self = tf.keras.layers.Dense(d_model)(dec_out3)
        k_dec_self = tf.keras.layers.Dense(d_model)(dec_out3)
        v_dec_self = tf.keras.layers.Dense(d_model)(dec_out3)

        # (batch_size, seq_len_q, d_model)
        concat_attention_dec_self = AttLayer(num_heads, d_model, batch_size)([q_dec_self, k_dec_self, v_dec_self])

        attn_dec_1 = tf.keras.layers.Dense(d_model)(concat_attention_dec_self)  # (batch_size, seq_len_q, d_model)

        attn_dec_1 = tf.keras.layers.Dropout(.01)(attn_dec_1)
        dec_out1 = attn_dec_1
        dec_out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attn_dec_1 + dec_out3)

        q_dec_ed = tf.keras.layers.Dense(d_model)(dec_out1)
        k_dec_ed = tf.keras.layers.Dense(d_model)(enc_out2)
        v_dec_ed = tf.keras.layers.Dense(d_model)(enc_out2)

        # (batch_size, seq_len_q, d_model)
        concat_attention_dec_ed = AttLayer(num_heads, d_model, batch_size)([q_dec_ed, k_dec_ed, v_dec_ed])

        attn_dec_2 = tf.keras.layers.Dense(d_model)(concat_attention_dec_ed)  # (batch_size, seq_len_q, d_model)

        attn_dec_2 = tf.keras.layers.Dropout(.01)(attn_dec_2)
        dec_out2 = attn_dec_2
        dec_out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attn_dec_2 + dec_out1)
        ffn_output_dec = tf.keras.layers.Dense(dff, activation='relu')(dec_out2)
        ffn_output_dec = tf.keras.layers.Dense(d_model)(ffn_output_dec)
        ffn_output_dec = tf.keras.layers.Dropout(.01)(ffn_output_dec)
        dec_out3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ffn_output_dec + dec_out2)
        dec_out3 = ffn_output_dec
    # (batch_size, tar_seq_len, target_vocab_size)
    predictions = tf.keras.layers.Dense(target_vocab_size)(dec_out3)

    model = tf.keras.Model(
        inputs=[src, tgt], outputs=predictions)

    return model