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

TEST_BATCH_SIZE = 1
LIMIT = 50

def load_img(path, label):
    image = images.load_img(path)
    return image, label

def main():
    data = labels.load_data(MOS_PATH, IMG_DIRPATH)
    img_paths = data[:,0].astype(str)
    mos = data[:,1].astype(np.float32)
    print(f"Detected {len(mos)} labels and {len(img_paths)} images")

    if LIMIT < len(mos):
        print(f"Limited images to {LIMIT}")
        mos = mos[:LIMIT]
        img_paths = img_paths[:LIMIT]
    
    if not os.path.isfile(MODEL_PATH):
        print("Fatal error: model could not be found")
        sys.exit(1)

    model = models.load_model(MODEL_PATH)
    print(f"Loaded model")

    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mos))
    dataset = dataset.map(load_img)
    dataset = dataset.batch(TEST_BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    predictions = model.predict(dataset).flatten()
    mse = (predictions - mos) ** 2
    
    for i in range(len(predictions)):
        print(f"{predictions[i]:.3f} {mos[i]:.3f} mse: {mse[i]:.2f}")

    print(f"mse: {np.mean(mse)}")

    print("Program finished")

if __name__ == '__main__':
    main()
