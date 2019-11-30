import numpy as np

MAX_LENGTH = 40


def test(transformer, test_data):
    transformer.evaluate(test_data)


def translate(transformer, data, prelim_encoder):
    tgt_start_token_list = [prelim_encoder.get_tgt_start_token()]
    tgt_end_token = prelim_encoder.get_tgt_end_token()
    pad_token_list = [0]

    pred_id = 1
    src, _ = data
    for seq_src in src:
        greek = prelim_encoder.decode_src_word(
            [i for i in seq_src if i < prelim_encoder.get_src_vocab_size() - 2])
        seq_pred = []

        seq_tgt = tgt_start_token_list + pad_token_list
        seq_src = np.expand_dims(seq_src, 0)
        sequence_length = 0
        while pred_id != tgt_end_token and sequence_length != MAX_LENGTH:
            sequence_length += 1

            seq_tgt_np = np.expand_dims(seq_tgt, 0)

            preds = transformer.predict([seq_src, seq_tgt_np])
            preds = preds[:, -1:, :]

            pred_id = np.argmax(preds, axis=-1).astype(np.int)[0]
            if pred_id < prelim_encoder.get_tgt_vocab_size() - 2:
                word = prelim_encoder.decode_tgt_word(pred_id)

                seq_pred.append(word)
                seq_tgt.insert(-1, pred_id[0])
            else:
                pass
                #print("This should mean stop token probably, so should end sentence.")

        print('Predicted translation: {}'.format(seq_pred))
