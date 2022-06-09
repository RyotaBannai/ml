import tensorflow as tf
from tensorflow import keras
from dataclasses import dataclass as dc
import numpy as np
from config import conf

# import matplotlib.pyplot as plt
from pprint import pprint as pp


@dc
class Dlnn:
    units: int
    categories: int
    inputs: np.array
    labels: np.array
    epochs: int = 100

    def __post_init__(self):
        print("\nTensorflow: Instantiation")

    def set_model(self):
        self.model = keras.Sequential()
        self.model.add(
            keras.layers.Dense(
                self.units, activation="relu", input_shape=(conf.PCA_DIMENSION,)
            )
        )
        self.model.add(keras.layers.Dense(self.categories, activation="softmax"))
        self.compile_model()

    def compile_model(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.RMSprop(0.01),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        print("Model Compiled.")

    def fit(self):
        history = self.model.fit(self.inputs, self.labels, epochs=self.epochs)
        print("\nhistory dict:", history.history)

    def evaluate_model(self, inputs, labels, verbose=2):
        test_loss, test_acc = self.model.evaluate(inputs, labels, verbose=verbose)
        print("\nTest accuracy:", test_acc)

    def predict(self, tests_inputs, answers):
        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        predictions = probability_model.predict(tests_inputs)
        for p, a in zip(predictions, answers):
            print("Prediction:", np.argmax(p), ", Answer:", np.argmax(a))
