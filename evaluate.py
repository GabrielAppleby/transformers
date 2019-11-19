import tensorflow as tf
from data_preprocessing.data_preprocessing_utils import create_masks
from constants import MAX_LENGTH

TRAIN_STEP_SIGNATURE = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


def translate(transformer, sentence, prelim_encoder):
    """
    Translates a sentence using the transformer and prelim encoder.
    :param transformer: The transformer.
    :param prelim_encoder: The prelim encoder.
    :param sentence: The sentence to translate.
    :return: None. Prints the translation.
    """
    result, attention_weights = evaluate(transformer, prelim_encoder, sentence)

    predicted_sentence = prelim_encoder.decode_trgt([i for i in result
                                                     if
                                                     i < prelim_encoder.get_trgt_vocab_size() - 2])

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

    # test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    #
    # @tf.function(input_signature=TRAIN_STEP_SIGNATURE)
    # def test_step(inp, tar):
    #     """
    #     Takes one step in the training process.
    #     :param transformer: The transformer to train.
    #     :param optimizer: The optimizer to use.
    #     :param inp: The input sentence.
    #     :param tar: The target sentence.
    #     :return: None. As a side effect the weights of the transformer are changed.
    #     """
    #     tar_inp = tar[:, :-1]
    #     tar_real = tar[:, 1:]
    #
    #     enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp,
    #                                                                      tar_inp)
    #
    #     predictions, _ = transformer(inp, tar_inp,
    #                                  True,
    #                                  enc_padding_mask,
    #                                  combined_mask,
    #                                  dec_padding_mask)
    #     test_accuracy(tar_real, predictions)
    #     return predictions
    #
    # for (test_batch, (test_inp, test_tar)) in enumerate(test_dataset):
    #     decoder_input = [prelim_encoder.get_trgt_start_token()]
    #     output = tf.expand_dims(decoder_input, 0)
    #     preds = test_step(test_inp, output)
    #     print(preds)
    #     print(preds[0])
    #     print(preds.shape)
    #     print(preds[0].shape)
    #     predicted_sentence = prelim_encoder.decode_trgt([i for i in preds
    #                                                      if
    #                                                      i < prelim_encoder.get_trgt_vocab_size() - 2])
    #     print(test_accuracy.result())
    #     print('Real words: {}'.format(test_tar))
    #     print('Predicted translation: {}'.format(predicted_sentence))


def evaluate(transformer, prelim_encoder, inp_sentence):
    """
    Evaluates a sentence turning an input sentence into a target embedding.
    :param transformer: The transformer.
    :param prelim_encoder: The encoder.
    :param inp_sentence: The input sentence.
    :return: The target embeddings and the attention weights.
    """
    start_token = [prelim_encoder.get_inpt_start_token()]
    end_token = [prelim_encoder.get_inpt_end_token()]

    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + prelim_encoder.encode_inpt(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [prelim_encoder.get_trgt_start_token()]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == prelim_encoder.get_trgt_end_token():
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights
