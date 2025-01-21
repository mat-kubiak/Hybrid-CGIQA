import os
from tensorflow import keras

def model_exists(path):
    return os.path.isfile(path)

def init_model(max_height, max_width, ratings):
    model = keras.Sequential([
        keras.layers.Input(shape=(max_height, max_width, 3)),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(1024),
        keras.layers.Dense(1024),
        keras.layers.Dense(512),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(ratings, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.00001, momentum=0.9),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
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
