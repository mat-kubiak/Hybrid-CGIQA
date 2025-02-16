import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

from ordinalcrossentropy import OrdinalCrossentropy
from nima import load_pretrained_nima

SEED = 23478
tf.random.set_seed(SEED)
random.seed(SEED)

DROPOUT_DENSE = 0.5
ACT_DENSE = 'relu'
ACT_CNN = 'leaky_relu'
L2_DENSE = l2(1e-3)
L2_CNN = l2(1e-6)

def _dense_blocks(input_layer, units):
    x = input_layer
    for u in units:
        x = layers.Dense(units=u, activation=ACT_DENSE, kernel_regularizer=L2_DENSE)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(DROPOUT_DENSE)(x)
    return x

def _multi_channel_attention(input):
    i_shape = keras.backend.int_shape(input)
    channels = i_shape[-1]

    rate = 8
    d1 = layers.Dense(units=channels // rate, activation='leaky_relu', kernel_regularizer=l2(1e-7), name='attention_dense1')
    d2 = layers.Dense(units=channels, kernel_regularizer=l2(1e-7), name='attention_dense2')

    pool = (i_shape[1], i_shape[2])

    # avg
    a = layers.AveragePooling2D(pool_size=pool, name='attention_avgpool')(input)
    a = layers.Flatten(name='attention_avgpool_flat')(a)
    a = d1(a)
    a = d2(a)

    # max
    m = layers.MaxPooling2D(pool_size=pool, name='attention_maxpool')(input)
    m = layers.Flatten(name='attention_maxpool_flat')(m)
    m = d1(m)
    m = d2(m)

    # final
    f = layers.Add(name='attention_add')([a, m])
    f = layers.Activation('sigmoid', name='attention_act')(f)

    f = layers.Reshape([1, 1, channels], name='attention_reshape')(f)
    f = layers.Multiply(name='attention_mul')([f, input])
    return f

def _adaptive_average_pool_2D(x, target_shape):
    _, ih, iw, _ = keras.backend.int_shape(x)
    oh, ow = target_shape

    pool_size = (ih // oh, iw // ow)
    if pool_size == (1, 1):
        return x

    return layers.AveragePooling2D(pool_size=pool_size, strides=pool_size, padding="valid")(x)

def _hidden_layers(input_layer):

    # nima route
    nima = load_pretrained_nima()

    names = [
        'global_average_pooling2d',
        'dense',
    ]
    output_layers = [nima.get_layer(name).output for name in names]

    nima = keras.Model(inputs=nima.input, outputs=output_layers, name="nima_backbone")

    for layer in nima.layers:
        layer.trainable = False

    # [0,255] -> [-1,1]
    n = layers.Rescaling(scale=1.0/127.5, offset=-1.0)(input_layer)
    n = nima(n)
    n = layers.Concatenate()(n)
    n = _dense_blocks(n, [256, 128, 64])

    # conv route
    effnet = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        include_preprocessing=True
    )

    names = [
        'block1a_project_activation',
        'block2b_add',
        'block3b_add',
        'block4c_add',
        'block5e_add',
        'block6h_add',
        'top_activation',
    ]
    output_layers = [effnet.get_layer(name).output for name in names]

    effnet = tf.keras.models.Model(inputs=effnet.input, outputs=output_layers, name='efficient_net_v2_backbone')

    cc = effnet(input_layer)
    cc = [_adaptive_average_pool_2D(c, (7,7)) for c in cc]

    c = layers.Concatenate(axis=-1)(cc)
    c_channels = keras.backend.int_shape(c)[-1]

    c = layers.Conv2D(c_channels // 4, (1,1), padding='same', activation=ACT_CNN, kernel_regularizer=L2_CNN)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Conv2D(c_channels // 4, (3,3), padding='same', activation=ACT_CNN, kernel_regularizer=L2_CNN)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Conv2D(c_channels // 4, (1,1), padding='same', activation=ACT_CNN, kernel_regularizer=L2_CNN)(c)
    c = layers.BatchNormalization()(c)

    c = _multi_channel_attention(c)
    c = layers.GlobalAveragePooling2D()(c)

    c = _dense_blocks(c, [256, 128, 64])

    # merge
    x = layers.Concatenate()([n, c])
    x = _dense_blocks(x, [1024, 128])

    return x

def init_model_continuous(height, width):
    input_layer = layers.Input(shape=(height, width, 3))
    hidden_layers = _hidden_layers(input_layer)
    output_layer = layers.Dense(units=1, activation='linear')(hidden_layers)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    initial_learning_rate = 1e-4  # Start higher
    decay_steps = 1000
    decay_rate = 0.9

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.MeanSquaredError(),
        metrics=[
            keras.metrics.RootMeanSquaredError(),
            keras.metrics.MeanAbsoluteError(),
        ]
    )

    return model

def init_model_categorical(height, width):
    input_layer = layers.Input(shape=(height, width, 3))
    preprocessed = layers.Lambda(lambda x: x*2 - 1.0, name="preprocessing")(input_layer)

    hidden_layers = _hidden_layers(preprocessed)
    output_layer = layers.Dense(units=41, activation='softmax')(hidden_layers)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=OrdinalCrossentropy(),
        metrics=["accuracy"]
    )

    return model

def init_model(height, width, is_categorical):
    if is_categorical:
        return init_model_categorical(height, width)
    return init_model_continuous(height, width)

def load_model(path):
    return keras.models.load_model(path)

def save_model(model, path):
    if model is None:
        raise ValueError("Model has not been created.")
    
    model.save(path)
