import tensorflow as tf

from constants import MAX_LENGTH


def filter_max_length(inpt, trgt):
    """
    Filters to a maximum sentence length.
    :param inpt: The input sentence.
    :param trgt: The target sentence.
    :return: None.
    """
    return tf.logical_and(tf.size(inpt) <= MAX_LENGTH, tf.size(trgt) <= MAX_LENGTH)


def create_padding_mask(seq):
    """
    Makes sure the model does not attend to the padding..
    :param seq: The sequence to mask.
    :return: A mask for the padding portions of the sequence.
    """
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


def preprocess_data_set(data, prelim_encoder, buffer_size, batch_size, cache=False):
    """
    Performs the preliminary encoding of the data set, filters by max length,
    shuffles the data, and creates padded batches.
    :param data: The tra
    :param prelim_encoder:
    :param cache:
    :return:
    """
    encoded_data = data.map(prelim_encoder.tf_encode)
    encoded_data = encoded_data.filter(filter_max_length)
    encoded_data = encoded_data.shuffle(buffer_size).padded_batch(
        batch_size, padded_shapes=([-1], [-1]), drop_remainder=True)
    if cache:
        encoded_data = encoded_data.cache()
        encoded_data = encoded_data.prefetch(tf.data.experimental.AUTOTUNE)

    return encoded_data
