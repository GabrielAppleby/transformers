import os
import numpy as np
from sklearn.model_selection import train_test_split

FILE_PATH_PREFIX = "data"
RAW_DATA_NAME = "grc-eng_copy.txt"
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
    full_text = read_file(RAW_DATA_NAME)
    train, val_and_test = train_test_split(full_text, test_size=0.30, random_state=2019)
    val, test = train_test_split(val_and_test, test_size=0.50, random_state=2019)
    train_src = [x.split(',')[0] for x in train]
    train_tgt = [x.split(',')[1] for x in train]
    val_src = [x.split(',')[0] for x in val]
    val_tgt = [x.split(',')[1] for x in val]
    test_src = [x.split(',')[0] for x in test]
    test_tgt = [x.split(',')[1] for x in test]

    np.savez(DATA_NPZ_NAME,
             train_src=train_src, train_tgt=train_tgt,
             val_src=val_src, val_tgt=val_tgt,
             test_src=test_src, test_tgt=test_tgt)


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