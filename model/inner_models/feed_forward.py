import tensorflow as tf


def feed_forward_network(d_model, dff):
    """
    Returns a two layer feed forward network comprised of two dense layers.
    :param d_model: The number of units in the first layer.
    :param dff: The number of units in the seconds layer
    :return: The two layer feed forward network.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])
