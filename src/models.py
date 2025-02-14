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

    rate = 8
    d1 = layers.Dense(units=channels // rate, activation='leaky_relu', kernel_regularizer=l2(1e-7))
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

    DROPOUT_DENSE = 0.5
    ACT_DENSE = 'relu'
    ACT_CNN = 'leaky_relu'
    L2_DENSE = l2(1e-3)
    L2_CNN = l2(1e-6)

    # nima route
    nima = load_pretrained_nima()
    nima = keras.Model(inputs=nima.input, outputs=nima.layers[-4].output, name="pretrained_NIMA")
    
    for layer in nima.layers:
        layer.trainable = False

    n = layers.Lambda(lambda x: x*2 - 1.0)(input_layer) # transform to MobileNet input range of (-1., 1.)
    n = nima(n)
    n = layers.GlobalAveragePooling2D()(n)

    n = layers.Dense(units=256, activation=ACT_DENSE, kernel_regularizer=L2_DENSE)(n)
    n = layers.BatchNormalization()(n)
    n = layers.Dropout(DROPOUT_DENSE)(n)

    n = layers.Dense(units=128, activation=ACT_DENSE, kernel_regularizer=L2_DENSE)(n)
    n = layers.BatchNormalization()(n)
    n = layers.Dropout(DROPOUT_DENSE)(n)

    # conv route
    effnet = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
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

    output_layers = []
    for name in names:
        output_layers.append(effnet.get_layer(name).output)

    effnet = tf.keras.models.Model(inputs=effnet.input, outputs=output_layers, name='EfficientV2Backbone')
    # effnet.trainable = False

    f_shape = 7
    cc = effnet(input_layer)
    cc2 = []
    
    for c in cc:
        cc2.append(AdaptiveAveragePooling2D(grid_size=f_shape)(c))

    c = layers.Concatenate(axis=-1)(cc2)
    c_channels = keras.backend.int_shape(c)[-1]

    c = layers.Conv2D(c_channels // 4, (1,1), padding='same', activation=ACT_CNN, kernel_regularizer=L2_CNN)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Conv2D(c_channels // 4, (3,3), padding='same', activation=ACT_CNN, kernel_regularizer=L2_CNN)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Conv2D(c_channels // 4, (1,1), padding='same', activation=ACT_CNN, kernel_regularizer=L2_CNN)(c)
    c = layers.BatchNormalization()(c)

    c = channel_attention(c)

    c = layers.Dense(units=256, activation=ACT_DENSE, kernel_regularizer=L2_DENSE)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Dropout(DROPOUT_DENSE)(c)

    c = layers.Dense(units=128, activation=ACT_DENSE, kernel_regularizer=L2_DENSE)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Dropout(DROPOUT_DENSE)(c)

    # merge
    x = layers.Concatenate()([n, c])

    x = layers.Dense(units=256, activation=ACT_DENSE, kernel_regularizer=L2_DENSE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_DENSE)(x)

    x = layers.Dense(units=32, activation=ACT_DENSE, kernel_regularizer=L2_DENSE)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_DENSE)(x)

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
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=keras.losses.Huber(delta=0.1),
        metrics=[
            keras.metrics.RootMeanSquaredError(),
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
