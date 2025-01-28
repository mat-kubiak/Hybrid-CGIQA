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
IMG_DIRPATH = f'{DATA_PATH}/images/train'

HEIGHT = None
WIDTH = None
IS_CATEGORICAL = None

TEST_BATCH_SIZE = 1
LIMIT = 20

def load_image(path, label):
    image = images.load_image(path, HEIGHT, WIDTH, False)
    return image, label

def main():
    global HEIGHT, WIDTH, IS_CATEGORICAL

    model = models.load_model(MODEL_PATH)
    print(f"Loaded model")

    HEIGHT, WIDTH = model.input_shape[1:3]
    print(f"Found dimensions: width: {WIDTH}, height: {HEIGHT}")
    IS_CATEGORICAL = model.output_shape[-1] == 41
    print(f"Detected model is categorical: {IS_CATEGORICAL}")

    img_paths = images.get_image_list(IMG_DIRPATH)
    mos = labels.load(MOS_PATH, IMG_DIRPATH, IS_CATEGORICAL)
    print(f"Detected {len(mos)} labels and {len(img_paths)} images")

    if LIMIT < len(mos):
        print(f"Limited images to {LIMIT}")
        mos = mos[:LIMIT]
        img_paths = img_paths[:LIMIT]
    
    if not os.path.isfile(MODEL_PATH):
        print("Fatal error: model could not be found")
        sys.exit(1)

    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mos))
    dataset = dataset.map(load_image)
    dataset = dataset.batch(TEST_BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    predictions = model.predict(dataset).flatten()
    mae = np.abs(predictions - mos)

    for i in range(len(predictions)):
        print(f"{predictions[i]*100:.1f} {mos[i]*100:.1f} {mae[i]:.2f}")
    
    print(f"mae: {np.mean(mae)}")

if __name__ == '__main__':
    main()
