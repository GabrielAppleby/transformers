import os
import tensorflow as tf
import numpy as np

FILE_PATH_PREFIX = "../data/data"
SRC_TRAIN = "src-train.txt"
TRGT_TRAIN = "tgt-train.txt"
TRAIN_FILES = [SRC_TRAIN, TRGT_TRAIN]
SRC_VAL = "src-val.txt"
TRGT_VAL = "tgt-val.txt"
VAL_FILES = [SRC_VAL, TRGT_VAL]

DATA_NPZ_NAME = "data.npz"


def main():
    """
    Trains the transformer and then translates a single sentence..
    :return: None
    """
    train_src = read_file(SRC_TRAIN)
    train_tgt = read_file(TRGT_TRAIN)
    val_src = read_file(SRC_VAL)
    val_tgt = read_file(TRGT_VAL)
    # val = read_files(VAL_FILES)
    np.savez(
        DATA_NPZ_NAME, train_src=train_src, train_tgt=train_tgt, val_src=val_src, val_tgt=val_tgt)


def read_file(file_name):
    with open(os.path.join(FILE_PATH_PREFIX, file_name), "r") as file:
        return file.readlines()


# def read_files(file_names):
#     texts = []
#     for file_name in file_names:
#         texts.append(read_file(file_name))
#     src_and_text = np.array(
#         [(line_src, line_trgt) for line_src, line_trgt in zip(texts[0], texts[1])])
#     return src_and_text


if __name__ == "__main__":
    main()
