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


def compile_model(transformer, d_model):
    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    transformer.compile(
        optimizer=optimizer,
        loss=masked_sparse_cross_entropy)
    return transformer
