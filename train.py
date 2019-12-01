import tensorflow as tf


def train(transformer, training_data, validation_data):
    transformer.fit(training_data,
                    epochs=999,
                    validation_data=validation_data,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                            patience=20, restore_best_weights=True)])
    return transformer
