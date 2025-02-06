import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

from histogram import NormalizedHistogram
from attention import SpatialAttention
from ordinalcrossentropy import OrdinalCrossentropy
from rgbtohsv import RGBToHSV

SEED = 23478
tf.random.set_seed(SEED)
random.seed(SEED)

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

def LightInceptionModule(x, filters_1x1, filters_3x3, filters_5x5):
    conv1x1 = layers.Conv2D(filters_1x1, (1,1), padding='same', activation='relu')(x)
    
    conv3x3 = layers.Conv2D(filters_3x3, (3,3), padding='same', activation='relu')(x)

    conv5x5 = layers.Conv2D(filters_5x5, (5,5), padding='same', activation='relu')(x)

    pool = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    pool_proj = layers.Conv2D(filters_1x1, (1,1), padding='same', activation='relu')(pool)
    
    return layers.Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5, pool_proj])

def _spp(x):
    spp_1 = layers.AveragePooling2D(pool_size=(x.shape[1], x.shape[2]))(x)
    spp_2 = layers.AveragePooling2D(pool_size=(x.shape[1]//2, x.shape[2]//2), strides=(x.shape[1]//2, x.shape[2]//2))(x)
    spp_4 = layers.AveragePooling2D(pool_size=(x.shape[1]//4, x.shape[2]//4), strides=(x.shape[1]//4, x.shape[2]//4))(x)
    spp_8 = layers.AveragePooling2D(pool_size=(x.shape[1]//8, x.shape[2]//8), strides=(x.shape[1]//8, x.shape[2]//8))(x)
    
    spp_1 = layers.Flatten()(spp_1)
    spp_2 = layers.Flatten()(spp_2)
    spp_4 = layers.Flatten()(spp_4)
    spp_8 = layers.Flatten()(spp_8)
    
    x = layers.Concatenate()([spp_1, spp_2, spp_4, spp_8])
    return x

def _hidden_layers(input_layer):
    
    # hist route
    h = RGBToHSV()(input_layer)
    h = NormalizedHistogram(nbins=256)(h)
    h = layers.Flatten()(h)
    h = layers.Dense(units=512, activation='relu', kernel_regularizer=l2(1e-4))(h)
    h = layers.Dropout(0.4, seed=seed)(h)

    # conv route
    c = layers.Conv2D(16, (7,7), strides=(2,2), padding='same', activation='relu')(input_layer)
    c = layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(c)
    
    c = LightInceptionModule(c, 8, 16, 8)
    c = LightInceptionModule(c, 16, 24, 16)
    c = layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(c)
    
    c = _spp(c)
    c = layers.Dense(units=256, activation="relu", kernel_regularizer=l2(1e-4))(c)
    c = layers.Dropout(0.4, seed=seed)(c)

    # merge
    x = layers.Concatenate()([h, c])
    x = layers.Dense(units=256, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.Dense(units=256, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.4, seed=seed)(x)

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
