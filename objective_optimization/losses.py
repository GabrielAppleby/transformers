import tensorflow as tf


def masked_sparse_cross_entropy(y_true, y_pred, *args, **kwards):
    y_true = y_true[:, 1:]
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))

    loss_object = tf.keras.losses. \
        SparseCategoricalCrossentropy(from_logits=True,
                                      reduction=tf.compat.v2.keras.losses.Reduction.NONE)
    loss_ = loss_object(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ = loss_ * mask
    loss_ = tf.squeeze(loss_)

    return tf.reduce_mean(loss_)
