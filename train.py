import tensorflow as tf


def train(transformer, training_data, validation_data):
    transformer.fit(training_data,
                    epochs=20,
                    validation_data=validation_data)
    return transformer
