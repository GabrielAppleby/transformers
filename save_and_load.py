import os
import json
import tensorflow as tf

from model.layers.full_encoder import Encoder
from model.layers.encoder_block import EncoderBlock
from model.layers.full_decoder import Decoder
from model.layers.decoder_block import DecoderBlock
from model.layers.multi_head_attention import MultiHeadAttention

MODELS_PATH = "saved_models"
MODEL_NAME = "transformer"
FILE_NAME = os.path.join(MODELS_PATH, MODEL_NAME) + '.json'


# In order to save models all custom objects must be announced to Keras.
CUSTOM_OBJECTS = {'Encoder': Encoder,
                  'Decoder': Decoder,
                  'EncoderBlock': EncoderBlock,
                  'DecoderBlock': DecoderBlock,
                  'MultiHeadAttention': MultiHeadAttention}


def save_model(model):
    model_json = model.to_json()
    with open(FILE_NAME, 'w', encoding='utf-8') as f:
        json.dump(model_json, f, ensure_ascii=False, indent=4)


def load_model():
    with open(FILE_NAME) as f:
        model_json = json.load(f)
    model = tf.keras.models.model_from_json(model_json, custom_objects=CUSTOM_OBJECTS)
    return model
