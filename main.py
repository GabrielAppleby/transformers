"""
Gabriel Appleby
"""

import tensorflow_datasets as tfds

from data_preprocessing.data_preprocessing_utils import preprocess_data_set, create_masks
from data_preprocessing.prelim_encoder import PrelimEncoder
from model.transformer_model import Transformer
from evaluate import translate
from train import train

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
    train_dataset, val_dataset, prelim_encoder = get_data()
    transformer = get_transformer(prelim_encoder)
    train(transformer, train_dataset, D_MODEL)
    translate(transformer, prelim_encoder, "este é um problema que temos que resolver.")
    print("Real translation: this is a problem we have to solve .")


def get_data():
    """
    Grabs and preprocesses the data.
    :return: training data, validation data, and the prelim_encoder.
    """
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                                   as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    prelim_encoder = PrelimEncoder(train_examples)
    val_dataset = preprocess_data_set(val_examples, prelim_encoder, BUFFER_SIZE, BATCH_SIZE)
    train_dataset = preprocess_data_set(
        train_examples, prelim_encoder, BUFFER_SIZE, BATCH_SIZE, cache=False)
    return train_dataset, val_dataset, prelim_encoder


def get_transformer(prelim_encoder):
    """
    Gets the transformer using the prelim encoder to set vocab sizes.
    :param prelim_encoder: The preliminary encoder of the texts.
    :return: The transformer.
    """
    input_vocab_size = prelim_encoder.get_inpt_vocab_size()
    target_vocab_size = prelim_encoder.get_trgt_vocab_size()

    return Transformer(input_vocab_size,
                       target_vocab_size,
                       input_vocab_size,
                       target_vocab_size,
                       num_blocks=NUM_LAYERS,
                       d_model=D_MODEL,
                       num_heads=NUM_HEADS,
                       dff=DFF,
                       drop_rate=DROPOUT_RATE)


if __name__ == "__main__":
    main()
