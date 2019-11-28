import tensorflow as tf

from model.layers.full_decoder import Decoder
from model.layers.full_encoder import Encoder


def get_transformer(input_vocab_size, target_vocab_size, num_blocks, d_model, num_heads, dff, drop_rate):
    src = tf.keras.layers.Input(shape=[None])
    tgt = tf.keras.layers.Input(shape=[None])
    enc_out = Encoder(num_blocks, d_model, num_heads, dff, input_vocab_size, drop_rate)(src)
    dec_out = Decoder(num_blocks, d_model, num_heads, dff, target_vocab_size, drop_rate)(
        [tgt, enc_out])
    predictions = tf.keras.layers.Dense(target_vocab_size)(dec_out)
    model = tf.keras.Model(inputs=[src, tgt], outputs=predictions)

    return model
