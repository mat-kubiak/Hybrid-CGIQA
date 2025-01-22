import os
import sys
import tensorflow as tf
import numpy as np

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{project_dir}/src')

import images, labels, models

# Path to the database containing the images and mos.csv
DATA_PATH = f'{project_dir}/data'

MODEL_PATH = f'{project_dir}/output/model.keras'
MOS_PATH = f'{DATA_PATH}/mos.csv'
IMG_DIRPATH = f'{DATA_PATH}/images/test'

MAX_HEIGHT = None
MAX_WIDTH = None
RATINGS = 41  # Range 1.0 to 5.0 with step 0.1 (41 distinct ratings)

TEST_BATCH_SIZE = 5

def load_img(path, label):
    image = images.load_img(path, MAX_HEIGHT, MAX_WIDTH)
    return image, label

def main():
    global MAX_HEIGHT, MAX_WIDTH
    mos = labels.load_labels(MOS_PATH, IMG_DIRPATH)
    print(f"Loaded {mos.shape[0]} labels")
    if mos.shape[0] == 0:
        print("Fatal error: no labels found")
        sys.exit(1)

    image_paths = images.get_image_list(IMG_DIRPATH)
    image_paths = IMG_DIRPATH + "/" + image_paths

    if not models.model_exists(MODEL_PATH):
        print("Fatal error: no model found")
        sys.exit(1)
    model = models.load_model(MODEL_PATH)
    MAX_HEIGHT = model.input_shape[1]
    MAX_WIDTH = model.input_shape[2]
    print(f"Found dimensions from model: width: {MAX_WIDTH}, height: {MAX_HEIGHT}")

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mos))
    dataset = dataset.map(load_img)
    dataset = dataset.batch(TEST_BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    loss, accuracy = model.evaluate(dataset)
    print(f"loss: {loss}")
    print(f"accuracy: {accuracy}")

    print("Program finished")

if __name__ == '__main__':
    main()
