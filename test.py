import numpy as np
import tensorflow as tf
from train import train
from model.transformer_model import get_transformer
from data_preprocessing.prelim_encoder import PrelimEncoder
from data_preprocessing.data_preprocessing_utils import preprocess_data_set

BUFFER_SIZE = 20000
BATCH_SIZE = 64
D_MODEL = 128
NUM_LAYERS = 4
DFF = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1


def main():
    """
    Trains the transformer and then translates a single sentence..
    :return: None
    """
    train_dataset, prelim_encoder, pad_shape = get_data()
    transformer = get_transformer_h(prelim_encoder, pad_shape)
    train(transformer, train_dataset, D_MODEL)

def get_data():
    """
    Grabs and preprocesses the data.
    :return: training data, validation data, and the prelim_encoder.
    """
    data = np.load("data.npz")
    raw_train_src = data["train_src"]
    raw_train_tgt = data["train_tgt"]
    raw_val_src = data["val_src"]
    raw_val_tgt = data["val_tgt"]
    raw_test_src = data["test_src"]
    raw_test_tgt = data["test_tgt"]

    # train_examples = tf.data.Dataset.from_tensor_slices((raw_train_src, raw_train_tgt))
    # val_examples = tf.data.Dataset.from_tensor_slices((raw_val_src, raw_val_tgt))
    # test_examples = tf.data.Dataset.from_tensor_slices((raw_test_src, raw_test_tgt))
    nonsense = (raw_train_src, raw_train_tgt)
    prelim_encoder = PrelimEncoder(nonsense)
    # val_dataset = preprocess_data_set(val_examples, prelim_encoder, BUFFER_SIZE, BATCH_SIZE)
    train_dataset, pad_shp = preprocess_data_set(
        nonsense, prelim_encoder, BUFFER_SIZE, BATCH_SIZE, cache=False)
    return train_dataset, prelim_encoder, pad_shp
    # test_dataset = preprocess_data_set(
    #     test_examples, prelim_encoder, BUFFER_SIZE, BATCH_SIZE, cache=False)
    # return train_dataset, val_dataset, raw_test_src[1], prelim_encoder

def get_transformer_h(prelim_encoder, pad_shape):
    """
    Gets the transformer using the prelim encoder to set vocab sizes.
    :param prelim_encoder: The preliminary encoder of the texts.
    :return: The transformer.
    """
    input_vocab_size = prelim_encoder.get_inpt_vocab_size()
    target_vocab_size = prelim_encoder.get_trgt_vocab_size()

    return get_transformer(input_vocab_size, target_vocab_size, input_vocab_size, target_vocab_size,
                       NUM_LAYERS,
                       D_MODEL,
                       NUM_HEADS,
                       DFF,
                       DROPOUT_RATE, pad_shape)

if __name__ == "__main__":
    main()
