import tensorflow as tf


def train(transformer, training_data, validation_data):
    transformer.fit(training_data,
                    epochs=20,
                    validation_data=validation_data,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                            patience=3, restore_best_weights=True)])
    return transformer
