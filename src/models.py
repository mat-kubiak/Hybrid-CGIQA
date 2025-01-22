import os
from tensorflow import keras

def model_exists(path):
    return os.path.isfile(path)

def init_model(max_height, max_width):

    model = keras.Sequential([
        keras.layers.Input(shape=(max_height, max_width, 3)),
        keras.layers.Conv2D(32, kernel_size=(5, 5), activation="relu", padding="valid"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(5, 5), activation="relu", padding="valid"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(128, kernel_size=(5, 5), activation="relu", padding="valid"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(256, kernel_size=(5, 5), activation="relu", padding="valid"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(1024),
        keras.layers.Dense(1024),
        keras.layers.Dense(512),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()]
    )

    return model

def train_model(model, x_train, y_train, epochs, batch_size):
    model.fit(
        x=x_train,
        y=y_train,
        verbose=1,
        epochs=epochs,
        batch_size=batch_size,
    )
    return model

def load_model(path):
    return keras.models.load_model(path)

def save_model(model, path):
    if model is None:
        raise ValueError("Model has not been created.")
    
    model.save(path)
