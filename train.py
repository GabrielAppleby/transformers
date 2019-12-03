import tensorflow as tf
from kerastuner.tuners import RandomSearch
from kerastuner import HyperParameters
from model.transformer_model import get_tuner_model


def train(transformer, training_data, validation_data):
    transformer.fit(training_data,
                    epochs=999,
                    validation_data=validation_data,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                            patience=3, restore_best_weights=True)])
    return transformer


def tune(train_data, validation_data, prelim_encoder):
    hp = HyperParameters()
    hp.Fixed('tgt_vocab_size', value=prelim_encoder.get_tgt_vocab_size())
    hp.Fixed('src_vocab_size', value=prelim_encoder.get_src_vocab_size())
    tuner = RandomSearch(
        get_tuner_model,
        objective='val_loss',
        hyperparameters=hp,
        max_trials=5,
        executions_per_trial=3,
        directory="tuner",
        project_name="transformer")
    tuner.search(train_data,
                 epochs=20,
                 validation_data=validation_data)
    return tuner.get_best_models(num_models=1)[0]

