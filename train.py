import time
import sys

import tensorflow as tf

from data_preprocessing.data_preprocessing_utils import create_masks
from objective_optimization.losses import loss_function

CHECKPOINT_PATH = "./checkpoints/train"
TRAIN_STEP_SIGNATURE = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


def train(transformer, data, d_model):
    """
    Trains the transformer on the train_dataset.
    :param transformer: The transformer to train.
    :param train_dataset: The train_dataset.
    :param d_model: The size of the embeddings.
    :return: None. As a side effect transformer is trained.
    """

    # Defines the training loss and accuracy
    # loss_object =

    # def loss_function(real, pred):
    #     # real = real[:, 1:]
    #     mask = tf.math.logical_not(tf.math.equal(real, 0))
    #     loss_ = loss_object(real, pred)
    #
    #     mask = tf.cast(mask, dtype=loss_.dtype)
    #     loss_ *= mask
    #     # test = tf.reduce_mean(loss_)
    #
    #     return loss_

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    def custom_loss(y_true, y_pred):
        return train_loss(y_true, y_pred)
    print(transformer.summary())
    transformer.compile(
        optimizer="Adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    transformer.fit(data, epochs=1)


# def train(transformer, train_dataset, val_dataset, d_model):
#     """
#     Trains the transformer on the train_dataset.
#     :param transformer: The transformer to train.
#     :param train_dataset: The train_dataset.
#     :param d_model: The size of the embeddings.
#     :return: None. As a side effect transformer is trained.
#     """
#
#     # Initializes an Adam optimzer
#     optimizer = tf.keras.optimizers.Adam(
#         CustomSchedule(d_model), beta_1=0.9, beta_2=0.98, epsilon=1e-9)
#
#     # Defines the training loss and accuracy
#     train_loss = tf.keras.metrics.Mean(name='train_loss')
#     val_loss = tf.keras.metrics.Mean(name='train_loss')
#     train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
#
#     # Initializes a Checkpoint
#     ckpt = tf.train.Checkpoint(transformer=transformer,
#                                optimizer=optimizer)
#
#     # Initializes a CheckpointManager
#     ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=5)
#
#     # if a checkpoint exists, restore the latest checkpoint.
#     if ckpt_manager.latest_checkpoint:
#         ckpt.restore(ckpt_manager.latest_checkpoint)
#         print('Latest checkpoint restored!!')
#
#     ###########################################################################
#     # The @tf.function trace-compiles train_step into a TF graph for faster
#     # execution. The function specializes to the precise shape of the argument
#     # tensors. To avoid re-tracing due to the variable sequence lengths or variable
#     # batch sizes (the last batch is smaller), use input_signature to specify
#     # more generic shapes.
#     # @tf.function(input_signature=TRAIN_STEP_SIGNATURE)
#     def train_step(inp, tar):
#         """
#         Takes one step in the training process.
#         :param transformer: The transformer to train.
#         :param optimizer: The optimizer to use.
#         :param inp: The input sentence.
#         :param tar: The target sentence.
#         :return: None. As a side effect the weights of the transformer are changed.
#         """
#         tar_inp = tar[:, :-1]
#         tar_real = tar[:, 1:]
#
#         enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp,
#                                                                          tar_inp)
#
#         with tf.GradientTape() as tape:
#             predictions, _ = transformer(inp, tar_inp,
#                                          True,
#                                          enc_padding_mask,
#                                          combined_mask,
#                                          dec_padding_mask)
#             loss = loss_function(tar_real, predictions)
#
#         gradients = tape.gradient(loss, transformer.trainable_variables)
#         optimizer.apply_gradients(
#             zip(gradients, transformer.trainable_variables))
#
#         train_loss(loss)
#         train_accuracy(tar_real, predictions)
#
#     @tf.function(input_signature=TRAIN_STEP_SIGNATURE)
#     def val_step(inp, tar):
#         """
#         Takes one step in the training process.
#         :param transformer: The transformer to train.
#         :param optimizer: The optimizer to use.
#         :param inp: The input sentence.
#         :param tar: The target sentence.
#         :return: None. As a side effect the weights of the transformer are changed.
#         """
#         tar_inp = tar[:, :-1]
#         tar_real = tar[:, 1:]
#
#         enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp,
#                                                                          tar_inp)
#
#         predictions, _ = transformer(inp, tar_inp,
#                                      True,
#                                      enc_padding_mask,
#                                      combined_mask,
#                                      dec_padding_mask)
#         loss = loss_function(tar_real, predictions)
#
#         val_loss(loss)
#     ###########################################################################
#
#     # Actual training loop
#     for epoch in range(1):
#         start = time.time()
#
#         train_loss.reset_states()
#         val_loss.reset_states()
#         train_accuracy.reset_states()
#
#         for (batch, (inp, tar)) in enumerate(train_dataset):
#             train_step(inp, tar)
#
#             if batch % 50 == 0:
#                 print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
#                     epoch + 1, batch, train_loss.result(), train_accuracy.result()))
#
#         if (epoch + 1) % 5 == 0:
#             ckpt_save_path = ckpt_manager.save()
#             print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
#                                                                 ckpt_save_path))
#
#             for (val_batch, (val_inp, val_tar)) in enumerate(val_dataset):
#                 val_step(val_inp, val_tar)
#                 print(val_loss.result())
#
#         print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
#                                                             train_loss.result(),
#                                                             train_accuracy.result()))
#
#         print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
