import tensorflow as tf

from model.layers.full_decoder import Decoder
from model.layers.full_encoder import Encoder
from objective_optimization.custom_schedule import CustomSchedule
from objective_optimization.losses import masked_sparse_cross_entropy


def get_transformer(input_vocab_size, target_vocab_size, num_blocks, d_model, num_heads, dff, drop_rate):
    src = tf.keras.layers.Input(shape=[None])
    tgt = tf.keras.layers.Input(shape=[None])
    enc_out = Encoder(num_blocks, d_model, num_heads, dff, input_vocab_size, drop_rate)(src)
    dec_out = Decoder(num_blocks, d_model, num_heads, dff, target_vocab_size, drop_rate)(
        [tgt, enc_out])
    predictions = tf.keras.layers.Dense(target_vocab_size)(dec_out)
    model = tf.keras.Model(inputs=[src, tgt], outputs=predictions)

    return model


def get_tuner_model(hp):
    num_blocks = hp.Int("num_blocks", min_value=1, max_value=4, step=1)
    d_model = hp.Int("d_model", min_value=64, max_value=128, step=32)
    tgt_vocab_size = hp.Int("tgt_vocab_size", min_value=1, max_value=2)
    src_vocab_size = hp.Int("src_vocab_size", min_value=1, max_value=2)
    num_heads_what = hp.Choice("num_heads_what", [2, 4, 8], default=2)
    dff = hp.Int("dff", min_value=128, max_value=512, step=64)
    drop_rate = hp.Float("drop_out", min_value=.1, max_value=.4, step=.05)

    src = tf.keras.layers.Input(shape=[None])
    tgt = tf.keras.layers.Input(shape=[None])
    enc_out = Encoder(num_blocks, d_model, num_heads_what, dff, src_vocab_size, drop_rate)(src)
    dec_out = Decoder(num_blocks, d_model, num_heads_what, dff, tgt_vocab_size, drop_rate)(
        [tgt, enc_out])
    predictions = tf.keras.layers.Dense(tgt_vocab_size)(dec_out)
    model = tf.keras.Model(inputs=[src, tgt], outputs=predictions)

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    model.compile(
        optimizer=optimizer,
        loss=masked_sparse_cross_entropy)
    return model


def compile_model(transformer, d_model):
    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    transformer.compile(
        optimizer=optimizer,
        loss=masked_sparse_cross_entropy)
    return transformer
