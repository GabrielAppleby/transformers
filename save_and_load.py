import os
import tensorflow as tf

from model.layers.full_encoder import Encoder
from model.layers.encoder_block import EncoderBlock
from model.layers.full_decoder import Decoder
from model.layers.decoder_block import DecoderBlock
from model.layers.multi_head_attention import MultiHeadAttention

MODELS_PATH = "saved_models"
MODEL_NAME = "transformer"
FILE_NAME = os.path.join(MODELS_PATH, MODEL_NAME) + ".h5"


# In order to save models all custom objects must be announced to Keras.
CUSTOM_OBJECTS = {'Encoder': Encoder,
                  'Decoder': Decoder,
                  'EncoderBlock': EncoderBlock,
                  'DecoderBlock': DecoderBlock,
                  'MultiHeadAttention': MultiHeadAttention}


def save_model(model):
    model.save(FILE_NAME, include_optimizer=False)


def load_model():
    return tf.keras.models.load_model(
        FILE_NAME, custom_objects=CUSTOM_OBJECTS, compile=False)
