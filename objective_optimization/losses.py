import tensorflow as tf


def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction=tf.compat.v2.keras.losses.Reduction.NONE)
    loss_ = loss_object(real, pred)

    return tf.reduce_mean(loss_)


class MaskedSparseCrossEntropy:

    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred, *args, **kwargs):
        # y_true = tf.boolean_mask(y_true, tf.equal(y_true, 0))
        # y_pred = tf.boolean_mask(y_pred, tf.equal(y_true, 0))
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                    reduction=tf.compat.v2.keras.losses.Reduction.NONE)
        loss_ = loss_object(y_true, y_pred)

        return tf.reduce_mean(loss_)
