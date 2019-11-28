import tensorflow as tf
from objective_optimization.losses import MaskedSparseCrossEntropy


def compile_and_train(transformer, training_data, validation_data):
    transformer = compile_model(transformer)
    transformer = train(transformer, training_data, validation_data)

    return transformer


def compile_model(transformer):
    transformer.compile(
        optimizer="Adam",
        loss=MaskedSparseCrossEntropy())
    return transformer


def train(transformer, training_data, validation_data):
    transformer.fit(training_data,
                    epochs=999,
                    validation_data=validation_data,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
    return transformer

