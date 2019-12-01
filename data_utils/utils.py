import numpy as np
from data_utils.prelim_encoder import PrelimEncoder
import tensorflow as tf
import tensorflow_datasets as tfds


def get_sanity_check_data():
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                   with_info=True,
                                   as_supervised=True)
    examples = tfds.as_numpy(examples, graph=None)

    train_src = []
    train_tgt = []
    for src, tgt in examples["train"]:
        train_src.append(src)
        train_tgt.append(tgt)

    return train_src, train_tgt


def get_npz_translation_data(file_path):
    data = np.load(file_path)
    raw_train_src = data["train_src"]
    raw_train_tgt = data["train_tgt"]
    raw_val_src = data["val_src"]
    raw_val_tgt = data["val_tgt"]
    raw_test_src = data["test_src"]
    raw_test_tgt = data["test_tgt"]

    raw_train = (raw_train_src, raw_train_tgt)
    raw_val = (raw_val_src, raw_val_tgt)
    raw_test = (raw_test_src, raw_test_tgt)
    return raw_train, raw_val, raw_test


def get_prelim_encoder(raw_training_data):
    return PrelimEncoder(raw_training_data)


def preprocess_data_set(data, prelim_encoder, buffer_size, batch_size, inference=False):
    src, tgt = prelim_encoder.encode(*data)
    src = tf.keras.preprocessing.sequence.pad_sequences(src, padding="post")
    tgt = tf.keras.preprocessing.sequence.pad_sequences(tgt, padding="post")

    if inference:
        data = (src, tgt)
    else:
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
