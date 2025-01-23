import os
import sys
import tensorflow as tf
import numpy as np

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{project_dir}/src')

import images, labels, models

MODEL_PATH = f'{project_dir}/output/model.keras'

DATA_PATH = f'{project_dir}/data'
MOS_PATH = f'{DATA_PATH}/mos.csv'
IMG_DIRPATH = f'{DATA_PATH}/images/test'

MAX_HEIGHT = None
MAX_WIDTH = None

TEST_BATCH_SIZE = 5
LIMIT = 50

def load_img(path, label):
    image = images.load_img(path, MAX_HEIGHT, MAX_WIDTH)
    return image, label

def main():
    global MAX_HEIGHT, MAX_WIDTH

    mos = labels.load_labels(MOS_PATH, IMG_DIRPATH)
    img_paths = images.get_image_list(IMG_DIRPATH)
    print(f"Detected {len(mos)} labels and {len(img_paths)} images")

    if LIMIT < len(mos):
        print(f"Limited images to {LIMIT}")
        mos = mos[:LIMIT]
        img_paths = img_paths[:LIMIT]

    if not models.model_exists(MODEL_PATH):
        print("Fatal error: no model found")
        sys.exit(1)
    
    model = models.load_model(MODEL_PATH)
    print(f"Loaded model")
    
    MAX_HEIGHT, MAX_WIDTH = model.input_shape[1:3]
    print(f"Found dimensions: width: {MAX_WIDTH}, height: {MAX_HEIGHT}")

    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mos))
    dataset = dataset.map(load_img)
    dataset = dataset.batch(TEST_BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    predictions = model.predict(dataset).flatten()
    
    for i in range(len(predictions)):
        print(f"{predictions[i]:.2f} {mos[i]:.2f}")

    mse = np.mean((predictions - mos) ** 2)
    print(f"mse: {mse}")

    print("Program finished")

if __name__ == '__main__':
    main()
