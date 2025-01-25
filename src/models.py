import os
from tensorflow import keras
import tensorflow as tf

import tensorflow as tf

class NormalizedHistogram(tf.keras.layers.Layer):
    def __init__(self, nbins=256, **kwargs):
        super().__init__(**kwargs)
        self.nbins = nbins

    def call(self, inputs):
        histograms = []

        for c in range(3):
            channel = inputs[..., c]

            bins = tf.linspace(0.0, 1.0, self.nbins + 1)

            hist = tf.histogram_fixed_width_bins(channel, value_range=[0.0, 1.0], nbins=self.nbins)
            hist = tf.cast(hist, tf.float32)
            hist = hist / tf.reduce_sum(hist)

            histograms.append(hist)

        histograms = tf.stack(histograms, axis=-1)
        return histograms

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.nbins, 3)
    
    def get_config(self):
        config = super().get_config()
        config.update({"nbins": self.nbins, 'trainable': False})
        return config

def init_model():

    input_layer = tf.keras.layers.Input(shape=(None, None, 3))

    x = NormalizedHistogram(nbins=256)(input_layer)
    x = tf.keras.layers.Dense(units=128, activation='relu')(x)
    
    output_layer = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()]
    )

    return model

def load_model(path):
    return keras.models.load_model(path)

def save_model(model, path):
    if model is None:
        raise ValueError("Model has not been created.")
    
    model.save(path)
