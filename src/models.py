import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class NormalizedHistogram(tf.keras.layers.Layer):
    def __init__(self, nbins=256, **kwargs):
        super().__init__(**kwargs)
        self.nbins = nbins

    def call(self, inputs):
        
        def op_per_image(image):
            histograms = []
            for c in range(3):
                channel = image[..., c]
                channel = tf.reshape(channel, [-1])

                hist = tf.histogram_fixed_width(channel, value_range=[0.0, 1.0], nbins=self.nbins)
                hist = tf.cast(hist, tf.float32)
                hist = hist / tf.reduce_sum(hist)

                histograms.append(hist)

            histograms = tf.stack(histograms, axis=-1)
            return histograms
        
        histograms = tf.map_fn(op_per_image, inputs)
        return histograms

    def compute_output_shape(self, input_shape):
        # input: (None, h, w, 3)
        # output: (None, 256, 3)
        return (input_shape[0], self.nbins, 3)
    
    def get_config(self):
        config = super().get_config()
        config.update({"nbins": self.nbins, 'trainable': False})
        return config

def _hidden_layers(input_layer):
    x = NormalizedHistogram(nbins=256)(input_layer)
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation='relu')(x)
    return x

def init_model_categorical(height, width, ratings):
    input_layer = layers.Input(shape=(height, width, 3))
    hidden_layers = _hidden_layers(input_layer)
    output_layer = layers.Dense(units=ratings, activation='softmax')(hidden_layers)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    return model

def init_model_continuous(height, width):
    input_layer = layers.Input(shape=(height, width, 3))
    hidden_layers = _hidden_layers(input_layer)
    output_layer = layers.Dense(units=1, activation='linear')(hidden_layers)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

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
