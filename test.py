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

def compute_accuracy(true_labels, predictions):
    true_labels_category = np.argmax(true_labels, axis=1)
    accuracy = np.mean(true_labels_category == predictions)
    return accuracy

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

    predictions = model.predict(dataset)
    predicted_categories = np.argmax(predictions, axis=1)
    predicted_categories = np.argmax(predictions, axis=1)

    accuracy = compute_accuracy(true_labels_category, predicted_categories)
    print(f"Classification Accuracy: {accuracy * 100:.2f}%")

    print("Program finished")

if __name__ == '__main__':
    main()
