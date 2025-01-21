import os
import sys
import tensorflow as tf
import numpy as np

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{project_dir}/src')

import images, labels, models

# Path to the database containing the images and mos.csv
DATA_PATH = f'{project_dir}/data'

MODEL_PATH = f'{project_dir}/model.keras'
MOS_PATH = f'{DATA_PATH}/mos.csv'
IMG_DIRPATH = f'{DATA_PATH}/images/test'

MAX_HEIGHT = 1080
MAX_WIDTH = 1920
RATINGS = 41  # Range 1.0 to 5.0 with step 0.1 (41 distinct ratings)

TEST_BATCH_SIZE = 5

def load_img(path, label):
    image = images.load_img(path, MAX_HEIGHT, MAX_WIDTH)
    return image, label

def compute_metrics(true_labels, predictions):
    # Convert one-hot labels to continuous values
    true_labels_continuous = np.argmax(true_labels, axis=1) * 0.1 + 1.0
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(true_labels_continuous - predictions))
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((true_labels_continuous - predictions) ** 2)
    # Calculate Pearson Correlation Coefficient
    correlation = np.corrcoef(true_labels_continuous, predictions)[0, 1]
    return mae, mse, correlation

def main():
    mos = labels.load_labels(MOS_PATH, IMG_DIRPATH)
    print(f"Loaded {mos.shape[0]} labels")
    if (mos.shape[0] == 0):
        print("Fatal error: no labels found")
        sys.exit(1)

    image_paths = images.get_image_list(IMG_DIRPATH)
    image_paths = IMG_DIRPATH + "/" + image_paths

    print(image_paths.shape)
    print(mos.shape)

    if not models.model_exists(MODEL_PATH):
        print("Fatal error: no model found")
        sys.exit(1)
    model = models.load_model(MODEL_PATH)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mos))
    dataset = dataset.map(load_img)
    dataset = dataset.batch(TEST_BATCH_SIZE)

    # Predict and collect results
    predictions = []
    true_labels = []
    for image_batch, label_batch in dataset:
        prediction_batch = model.predict(image_batch)
        predictions.extend(prediction_batch)
        true_labels.extend(label_batch.numpy())

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Evaluate metrics
    mae, mse, correlation = compute_metrics(true_labels, predictions)
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Pearson Correlation Coefficient: {correlation}")

    print("Program finished")

if __name__ == '__main__':
    main()
