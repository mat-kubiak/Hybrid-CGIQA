import os
import sys
import tensorflow as tf
import numpy as np

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{project_dir}/src')

import images, labels, models

MODEL_NAME = ''
MODEL_PATH = f'{project_dir}/output/{MODEL_NAME}/model.keras'

DATA_PATH = f'{project_dir}/data'
MOS_PATH = f'{DATA_PATH}/mos.csv'
IMG_DIRPATH = f'{DATA_PATH}/images/test'

HEIGHT = None
WIDTH = None
IS_CATEGORICAL = None

TEST_BATCH_SIZE = 1
LIMIT = None
PRINT_LIMIT = 10

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

    if LIMIT != None and LIMIT < len(mos):
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

    if IS_CATEGORICAL:
        predictions = model.predict(dataset)
        predictions = np.argmax(predictions, axis=1)
        mos = np.argmax(mos, axis=1)

        accuracy = [predictions[i] == mos[i] for i in range(len(predictions))]
        accuracy = np.sum(accuracy) / len(predictions) * 100

        for i in range(min(PRINT_LIMIT, len(predictions))):
            print(f"{predictions[i]} {mos[i]}")

        print(f"accuracy: {accuracy:.2f}%")

    else:
        predictions = model.predict(dataset).flatten()
        mae = np.abs(predictions - mos)
        
        print(f"mae: {np.mean(mae)}")
        print(f"highest error: {np.max(mae)}")
        
        merged = np.stack((predictions, mos, mae, img_paths), axis=-1)
        indices = np.argsort(merged[:, 2].astype(float))[::-1]
        merged = merged[indices]

        print(f"Top {PRINT_LIMIT} predictions with highest error:")
        for i in range(PRINT_LIMIT):
            print(f"{i+1}. {float(merged[i,0]):.3f} for {float(merged[i,1]):.3f} (error {float(merged[i,2]):.4f}): {merged[i,3]}")

if __name__ == '__main__':
    main()
