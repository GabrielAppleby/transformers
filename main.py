from train import compile_and_train
from model.transformer_model import get_transformer
from data_utils.utils import preprocess_data_set, \
    get_prelim_encoder, get_npz_translation_data

BUFFER_SIZE = 20000
BATCH_SIZE = 64
D_MODEL = 128
NUM_BLOCKS = 4
D_FF = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1
HOMER_FILE_PATH = "data/data.npz"


def main():
    raw_train_data, raw_validation_data = get_npz_translation_data(HOMER_FILE_PATH)
    prelim_encoder = get_prelim_encoder(raw_train_data)
    train_data = preprocess_data_set(
        raw_train_data, prelim_encoder, BUFFER_SIZE, BATCH_SIZE)
    validation_data = preprocess_data_set(
        raw_validation_data, prelim_encoder, BUFFER_SIZE, BATCH_SIZE)

    transformer = get_transformer(prelim_encoder.get_src_vocab_size(),
                                  prelim_encoder.get_tgt_vocab_size(),
                                  NUM_BLOCKS,
                                  D_MODEL,
                                  NUM_HEADS,
                                  D_FF,
                                  DROPOUT_RATE)

    transformer = compile_and_train(transformer, train_data, validation_data)


if __name__ == "__main__":
    main()
