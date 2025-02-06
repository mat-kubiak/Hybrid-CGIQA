import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

from histogram import NormalizedHistogram
from attention import SpatialAttention
from ordinalcrossentropy import OrdinalCrossentropy
from rgbtohsv import RGBToHSV

def _squeeze_excite(input_layer, reduction_ratio=6):
    squeeze = layers.GlobalAveragePooling2D()(input_layer)

    reduced_channels = squeeze.shape[-1]

    x = layers.Dense(reduced_channels // reduction_ratio, activation="relu")(squeeze)
    x = layers.Dense(reduced_channels, activation="sigmoid")(x)
    
    x = layers.Reshape((1, 1, reduced_channels))(x)
    x = layers.Multiply()([input_layer, x])

    return x

def get_augmentation_model():
    model = keras.Sequential([
        layers.RandomRotation(0.002),
        layers.RandomTranslation(0.02, 0.02),
        layers.RandomZoom(0.02),
        
        layers.RandomBrightness(0.02, value_range=(0, 1)),
        layers.RandomContrast(0.02),
    ])

    return model

def get_sample_weights(labels, power=1.0):
    mean = tf.reduce_mean(labels)
    std = tf.math.reduce_std(labels)
    
    z_scores = tf.abs(labels - mean) / std
    
    weights = 1.0 + tf.pow(z_scores, power)
    
    weights = weights / tf.reduce_mean(weights)
    return weights

def _hidden_layers(input_layer):
    
    # hist route
    h = RGBToHSV()(input_layer)
    h = NormalizedHistogram(nbins=256)(h)
    h = layers.Flatten()(h)
    h = layers.Dense(units=512, activation='relu', kernel_regularizer=l2(1e-4))(h)
    h = layers.Dropout(0.4)(h)

    # conv route
    c = layers.Conv2D(kernel_size=(3,3), filters=32, kernel_regularizer=l2(1e-5))(input_layer)
    c = layers.Conv2D(kernel_size=(3,3), filters=32, kernel_regularizer=l2(1e-5))(c)
    c = SpatialAttention()(c)
    c = layers.AveragePooling2D(pool_size=(3,3))(c)
    c = layers.Conv2D(kernel_size=(3,3), filters=64, kernel_regularizer=l2(1e-5))(c)
    c = layers.Conv2D(kernel_size=(3,3), filters=64, kernel_regularizer=l2(1e-5))(c)
    c = SpatialAttention()(c)
    c = layers.AveragePooling2D(pool_size=(3,3))(c)

    c = layers.Conv2D(kernel_size=(3,3), filters=128, kernel_regularizer=l2(1e-5))(c)
    c = layers.Conv2D(kernel_size=(3,3), filters=128, kernel_regularizer=l2(1e-5))(c)
    c = SpatialAttention()(c)
    c = layers.AveragePooling2D(pool_size=(3,3))(c)

    c = layers.Conv2D(kernel_size=(1,1), filters=1, activation="linear", kernel_regularizer=l2(1e-5))(c)
    c = layers.Flatten()(c)

    c = layers.Dense(units=512, activation="relu", kernel_regularizer=l2(1e-4))(c)
    c = layers.Dropout(0.4)(c)

    # merge
    x = layers.Concatenate()([h, c])
    x = layers.Dense(units=512, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.Dense(units=256, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)

    return x

def init_model_continuous(height, width, gaussian=0):
    input_layer = layers.Input(shape=(height, width, 3))
    hidden_layers = _hidden_layers(input_layer)
    output_layer = layers.Dense(units=1, activation='linear')(hidden_layers)
    
    if gaussian != 0:
        output_layer = layers.GaussianNoise(gaussian)(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.MeanSquaredError(),
        # loss=keras.losses.Huber(delta=0.1),
        # loss=keras.losses.MeanAbsoluteError(),
        metrics=[keras.metrics.MeanAbsoluteError()]
    )

    return model

def init_model_categorical(height, width):
    input_layer = layers.Input(shape=(height, width, 3))
    hidden_layers = _hidden_layers(input_layer)
    output_layer = layers.Dense(units=41, activation='softmax')(hidden_layers)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=OrdinalCrossentropy(),
        metrics=["accuracy"]
    )

    return model

def init_model(height, width, is_categorical, gaussian=0):
    if is_categorical:
        return init_model_categorical(height, width)
    return init_model_continuous(height, width, gaussian=gaussian)

def load_model(path):
    return keras.models.load_model(path)

def save_model(model, path):
    if model is None:
        raise ValueError("Model has not been created.")
    
    model.save(path)
