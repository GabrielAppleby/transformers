import tensorflow as tf
from train import train
from test import test, translate
from model.transformer_model import get_transformer, compile_model
from data_utils.utils import preprocess_data_set, \
    get_prelim_encoder, get_npz_translation_data
from save_and_load import save_model, load_model

BUFFER_SIZE = 20000
BATCH_SIZE = 64
D_MODEL = 128
NUM_BLOCKS = 4
D_FF = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1
HOMER_FILE_PATH = "data/data.npz"


def main():
    tf.compat.v1.disable_eager_execution()
    raw_train_data, raw_validation_data, raw_test_data = get_npz_translation_data(HOMER_FILE_PATH)
    prelim_encoder = get_prelim_encoder(raw_train_data)

    train_data = preprocess_data_set(
        raw_train_data, prelim_encoder, BUFFER_SIZE, BATCH_SIZE)
    validation_data = preprocess_data_set(
        raw_validation_data, prelim_encoder, BUFFER_SIZE, BATCH_SIZE)
    test_data = preprocess_data_set(
        raw_test_data, prelim_encoder, BUFFER_SIZE, BATCH_SIZE, inference=True)
    transformer = load_model()
    # transformer = get_transformer(prelim_encoder.get_src_vocab_size(),
    #                               prelim_encoder.get_tgt_vocab_size(),
    #                               NUM_BLOCKS,
    #                               D_MODEL,
    #                               NUM_HEADS,
    #                               D_FF,
    #                               DROPOUT_RATE)
    # transformer = compile_model(transformer, D_MODEL)
    # transformer = train(transformer, train_data, validation_data)
    # save_model(transformer)
    translate(transformer, test_data, prelim_encoder)


if __name__ == "__main__":
    main()
