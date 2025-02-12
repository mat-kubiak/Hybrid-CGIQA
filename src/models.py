import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

from ordinalcrossentropy import OrdinalCrossentropy
from adaptivepooling import AdaptiveAveragePooling2D
from nima import load_pretrained_nima
from vendor.utils.losses import earth_movers_distance

SEED = 23478
tf.random.set_seed(SEED)
random.seed(SEED)

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

def channel_attention(input):
    i_shape = keras.backend.int_shape(input)
    channels = i_shape[-1]

    d1 = layers.Dense(units=64, activation='leaky_relu', kernel_regularizer=l2(1e-7))
    d2 = layers.Dense(units=channels, kernel_regularizer=l2(1e-7))

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

    reg = l2(1e-6)
    act_fn = 'leaky_relu'
    dropout = 0.1

    c = layers.Conv2D(32, (3,3), padding='same', activation=act_fn, kernel_regularizer=reg)(input_layer)
    c = layers.Conv2D(32, (3,3), padding='same', activation=act_fn, kernel_regularizer=reg)(c)
    c = layers.Dropout(dropout, seed=SEED)(c)
    cc.append(AdaptiveAveragePooling2D(grid_size=f_shape)(c))

    c = layers.AveragePooling2D(pool_size=(2,2))(c)
    c = layers.Conv2D(48, (3,3), padding='same', activation=act_fn, kernel_regularizer=reg)(c)
    c = layers.Conv2D(48, (3,3), padding='same', activation=act_fn, kernel_regularizer=reg)(c)
    c = layers.Dropout(dropout, seed=SEED)(c)
    cc.append(AdaptiveAveragePooling2D(grid_size=f_shape)(c))

    c = layers.AveragePooling2D(pool_size=(2,2))(c)
    c = layers.Conv2D(64, (3,3), padding='same', activation=act_fn, kernel_regularizer=reg)(c)
    c = layers.Conv2D(64, (3,3), padding='same', activation=act_fn, kernel_regularizer=reg)(c)
    c = layers.Dropout(dropout, seed=SEED)(c)
    cc.append(AdaptiveAveragePooling2D(grid_size=f_shape)(c))

    c = layers.AveragePooling2D(pool_size=(2,2))(c)
    c = layers.Conv2D(96, (3,3), padding='same', activation=act_fn, kernel_regularizer=reg)(c)
    c = layers.Conv2D(96, (3,3), padding='same', activation=act_fn, kernel_regularizer=reg)(c)
    c = layers.Dropout(dropout, seed=SEED)(c)
    cc.append(AdaptiveAveragePooling2D(grid_size=f_shape)(c))

    c = layers.AveragePooling2D(pool_size=(2,2))(c)
    c = layers.Conv2D(128, (3,3), padding='same', activation=act_fn, kernel_regularizer=reg)(c)
    c = layers.Conv2D(128, (3,3), padding='same', activation=act_fn, kernel_regularizer=reg)(c)
    c = layers.Dropout(dropout, seed=SEED)(c)
    cc.append(AdaptiveAveragePooling2D(grid_size=f_shape)(c))

    c = layers.AveragePooling2D(pool_size=(2,2))(c)
    c = layers.Conv2D(192, (3,3), padding='same', activation=act_fn, kernel_regularizer=reg)(c)
    c = layers.Conv2D(192, (3,3), padding='same', activation=act_fn, kernel_regularizer=reg)(c)
    c = layers.Dropout(dropout, seed=SEED)(c)
    cc.append(AdaptiveAveragePooling2D(grid_size=f_shape)(c))

    c = layers.AveragePooling2D(pool_size=(2,2))(c)
    c = layers.Conv2D(256, (3,3), padding='same', activation=act_fn, kernel_regularizer=reg)(c)
    c = layers.Conv2D(256, (3,3), padding='same', activation=act_fn, kernel_regularizer=reg)(c)
    c = layers.Dropout(dropout, seed=SEED)(c)
    cc.append(AdaptiveAveragePooling2D(grid_size=f_shape)(c))

    c = layers.Concatenate(axis=-1)(cc)
    c_channels = keras.backend.int_shape(c)[-1]

    c = layers.Conv2D(c_channels // 4, (1,1), padding='same', activation=act_fn, kernel_regularizer=reg)(c)
    c = layers.Conv2D(c_channels // 4, (3,3), padding='same', activation=act_fn, kernel_regularizer=reg)(c)
    c = layers.Conv2D(c_channels // 4, (1,1), padding='same', activation=act_fn, kernel_regularizer=reg)(c)

    c = channel_attention(c)

    # merge
    x = layers.Concatenate()([n, c])
    x = layers.Dense(units=1024, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.Dense(units=128, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)

    return x

@keras.utils.register_keras_serializable()
def emd(y_true, y_pred):
    return earth_movers_distance(y_true, y_pred)

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
        metrics=[
            keras.metrics.MeanSquaredError(),
            keras.metrics.MeanAbsoluteError(),
            emd
        ]
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
