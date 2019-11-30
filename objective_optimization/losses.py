import tensorflow as tf


class MaskedSparseCrossEntropy:

    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred, *args, **kwargs):
        y_true = y_true[:, 1:]
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))

        loss_object = tf.keras.losses.\
            SparseCategoricalCrossentropy(from_logits=True,
                                          reduction=tf.compat.v2.keras.losses.Reduction.NONE)
        loss_ = loss_object(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ = loss_ * mask

        return loss_
