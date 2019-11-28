import tensorflow as tf
from objective_optimization.losses import loss_function, MaskedSparseCrossEntropy


def compile_and_train(transformer, training_data):
    transformer = compile_model(transformer)
    transformer = train(transformer, training_data)

    return transformer


def compile_model(transformer):
    transformer.compile(
        optimizer="Adam",
        loss=MaskedSparseCrossEntropy())
    return transformer


def train(transformer, training_data):
    transformer.fit(training_data, epochs=1)
    return transformer

