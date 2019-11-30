import tensorflow as tf


def train(transformer, training_data, validation_data):
    transformer.fit(training_data,
                    epochs=999,
                    validation_data=validation_data,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
    return transformer

