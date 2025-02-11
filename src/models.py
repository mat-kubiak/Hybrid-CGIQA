import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

from ordinalcrossentropy import OrdinalCrossentropy
from adaptivepooling import AdaptiveAveragePooling2D
from nima import load_pretrained_nima

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
    conv1x1 = layers.Conv2D(filters_1x1, (1,1), padding='same', activation='relu', kernel_regularizer=l2(1e-5))(x)
    
    conv3x3 = layers.Conv2D(filters_3x3, (3,3), padding='same', activation='relu', kernel_regularizer=l2(1e-5))(x)

    conv5x5 = layers.Conv2D(filters_5x5, (5,5), padding='same', activation='relu', kernel_regularizer=l2(1e-5))(x)

    pool = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    pool_proj = layers.Conv2D(filters_1x1, (1,1), padding='same', activation='relu', kernel_regularizer=l2(1e-5))(pool)
    
    return layers.Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5, pool_proj])

def _spp(x):
    spp_1 = AdaptiveAveragePooling2D(grid_size=1)(x)
    spp_2 = AdaptiveAveragePooling2D(grid_size=2)(x)
    spp_4 = AdaptiveAveragePooling2D(grid_size=4)(x)
    spp_8 = AdaptiveAveragePooling2D(grid_size=8)(x)

    spp_1 = layers.Flatten()(spp_1)
    spp_2 = layers.Flatten()(spp_2)
    spp_4 = layers.Flatten()(spp_4)
    spp_8 = layers.Flatten()(spp_8)
    
    x = layers.Concatenate()([spp_1, spp_2, spp_4, spp_8])
    return x

def channel_attention(input):
    i_shape = keras.backend.int_shape(input)
    channels = i_shape[-1]

    d1 = layers.Dense(units=64)
    d2 = layers.Dense(units=channels)

    pool = (i_shape[1], i_shape[2])

    # avg
    a = layers.AveragePooling2D(pool_size=pool)(input)
    a = d1(a)
    a = d2(a)

    # max
    m = layers.MaxPooling2D(pool_size=pool)(input)
    m = d1(m)
    m = d2(m)

    # final
    f = layers.Add()([a, m])
    f = layers.Activation('sigmoid')(f)

    f = layers.Reshape([1, 1, channels])(f)
    f = layers.Multiply()([f, input])
    f = layers.AveragePooling2D(pool_size=pool)(f)
    f = layers.Flatten()(f)
    return f

def _hidden_layers(input_layer):

    # nima route
    nima = load_pretrained_nima()
    nima = keras.Model(inputs=nima.input, outputs=nima.layers[-4].output, name="pretrained_NIMA")
    
    for layer in nima.layers:
        layer.trainable = False

    n = layers.Resizing(224, 224, pad_to_aspect_ratio=True,)(input_layer)
    # n = AdaptiveAveragePooling2D(224)(input_layer)
    n = nima(n)
    n = layers.GlobalAveragePooling2D()(n)

    # conv route
    cc = []
    f_shape = 7

    c = layers.Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=l2(1e-5))(input_layer)
    c = layers.Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=l2(1e-5))(c)
    cc.append(AdaptiveAveragePooling2D(grid_size=f_shape)(c))

    c = layers.AveragePooling2D(pool_size=(2,2))(c)
    c = layers.Conv2D(48, (3,3), padding='same', activation='relu', kernel_regularizer=l2(1e-5))(c)
    c = layers.Conv2D(48, (3,3), padding='same', activation='relu', kernel_regularizer=l2(1e-5))(c)
    cc.append(AdaptiveAveragePooling2D(grid_size=f_shape)(c))

    c = layers.AveragePooling2D(pool_size=(2,2))(c)
    c = layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=l2(1e-5))(c)
    c = layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=l2(1e-5))(c)
    cc.append(AdaptiveAveragePooling2D(grid_size=f_shape)(c))

    c = layers.AveragePooling2D(pool_size=(2,2))(c)
    c = layers.Conv2D(96, (3,3), padding='same', activation='relu', kernel_regularizer=l2(1e-5))(c)
    c = layers.Conv2D(96, (3,3), padding='same', activation='relu', kernel_regularizer=l2(1e-5))(c)
    cc.append(AdaptiveAveragePooling2D(grid_size=f_shape)(c))

    c = layers.AveragePooling2D(pool_size=(2,2))(c)
    c = layers.Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=l2(1e-5))(c)
    c = layers.Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=l2(1e-5))(c)
    cc.append(AdaptiveAveragePooling2D(grid_size=f_shape)(c))

    c = layers.Concatenate(axis=-1)(cc)
    c_channels = keras.backend.int_shape(c)[-1]

    c = layers.Conv2D(c_channels // 4, (1,1), padding='same', activation='relu', kernel_regularizer=l2(1e-5))(c)
    c = layers.Conv2D(c_channels // 4, (3,3), padding='same', activation='relu', kernel_regularizer=l2(1e-5))(c)
    c = layers.Conv2D(c_channels // 4, (1,1), padding='same', activation='relu', kernel_regularizer=l2(1e-5))(c)

    c = channel_attention(c)

    # merge
    x = layers.Concatenate()([n, c])
    x = layers.Dense(units=1024, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.Dense(units=128, activation="relu", kernel_regularizer=l2(1e-4))(x)
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
