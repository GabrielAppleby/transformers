import numpy as np
from data_utils.prelim_encoder import PrelimEncoder
import tensorflow as tf


def get_npz_translation_data(file_path):
    data = np.load(file_path)
    raw_train_src = data["train_src"][:128]
    raw_train_tgt = data["train_tgt"][:128]
    # raw_val_src = data["val_src"]
    # raw_val_tgt = data["val_tgt"]
    # raw_test_src = data["test_src"]
    # raw_test_tgt = data["test_tgt"]

    raw_train = (raw_train_src, raw_train_tgt)
    return raw_train


def get_prelim_encoder(raw_training_data):
    return PrelimEncoder(raw_training_data)


def preprocess_data_set(data, prelim_encoder, buffer_size, batch_size):
    src, tgt = prelim_encoder.encode(*data)
    src = tf.keras.preprocessing.sequence.pad_sequences(src, padding="post")
    tgt = tf.keras.preprocessing.sequence.pad_sequences(tgt, padding="post")

    data = tf.data.Dataset.from_tensor_slices((src, tgt))

    data = data.shuffle(buffer_size).padded_batch(
        batch_size, padded_shapes=([None], [None]), drop_remainder=True)

    data = data.map(lambda x, y: ((x, y), y))

    return data


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)]


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask
