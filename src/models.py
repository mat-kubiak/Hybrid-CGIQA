import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from histogram import NormalizedHistogram
from attention import SpatialAttention

def _hidden_layers(input_layer):
    
    # histogram path
    h = NormalizedHistogram(nbins=256)(input_layer)
    h = layers.Flatten()(h)

    # convolution path
    c = layers.Conv2D(filters=32, kernel_size=(7,7))(input_layer)
    c = SpatialAttention()(c)
    c = layers.MaxPooling2D(pool_size=(3,3))(c)
    c = layers.Conv2D(filters=64, kernel_size=(7,7))(c)
    c = layers.GlobalMaxPooling2D()(c)

    # final path
    x = layers.Concatenate()([h, c])
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
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
