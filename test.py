import os, sys, traceback
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{PROJECT_DIR}/src')

import images, labels, models
from vendor.utils import losses

MODEL_NAME = ''
MODEL_PATH = f'{PROJECT_DIR}/output/{MODEL_NAME}/model.keras'

DATA_PATH = f'{PROJECT_DIR}/data'
MOS_PATH = f'{DATA_PATH}/mos.csv'
IMG_DIRPATH = f'{DATA_PATH}/images/test'

HEIGHT = None
WIDTH = None
IS_CATEGORICAL = None

BATCH_SIZE = 32
LIMIT = None
PRINT_LIMIT = 10

tf.keras.config.enable_unsafe_deserialization()

def logistic_function(y, beta1, beta2, beta3, beta4, beta5):
    out = beta1 * (0.5 - 1 / (1 + np.exp(beta2 * (y - beta3)))) + beta4 * y + beta5
    return out.astype(np.float32)

def load_image(path, label):
    image = images.load_image(path, HEIGHT, WIDTH)
    return image, label

def main():
    global HEIGHT, WIDTH, IS_CATEGORICAL

    if not os.path.isfile(MODEL_PATH):
        print("Fatal error: model could not be found")
        sys.exit(-1)

    try:
        model = models.load_model(MODEL_PATH)
        print(f"Loaded model")
    except Exception as e:
        print(f"Fatal error while initializing model")
        traceback.print_exc()
        sys.exit(-1)

    HEIGHT, WIDTH = model.input_shape[1:3]
    print(f"Found dimensions: width: {WIDTH}, height: {HEIGHT}")

    IS_CATEGORICAL = model.output_shape[-1] == 41
    type_text = 'classification' if IS_CATEGORICAL else 'regression' 
    print(f"Detected model is a: {type_text} model")

    img_paths = images.get_image_list(IMG_DIRPATH)
    mos = labels.load(MOS_PATH, IMG_DIRPATH, IS_CATEGORICAL)
    print(f"Detected {len(mos)} labels and {len(img_paths)} images")

    if LIMIT != None and LIMIT < len(mos):
        print(f"Limited images to {LIMIT}")
        mos = mos[:LIMIT]
        img_paths = img_paths[:LIMIT]

    dataset = (tf.data.Dataset.from_tensor_slices((img_paths, mos))
        .map(load_image)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

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

        params, covariance = curve_fit(logistic_function, predictions, mos, p0=[1, 1, 1, 1, 1])
        beta1, beta2, beta3, beta4, beta5 = params

        print("Fitted Parameters:")
        print(f"β1 = {beta1}, β2 = {beta2}, β3 = {beta3}, β4 = {beta4}, β5 = {beta5}")

        predictions = logistic_function(predictions, *params)

        mae = tf.keras.metrics.MeanAbsoluteError()
        mse = tf.keras.metrics.MeanSquaredError()
        rmse = tf.keras.metrics.RootMeanSquaredError()
        
        mae.update_state(mos, predictions)
        mse.update_state(mos, predictions)
        rmse.update_state(mos, predictions)

        mae = mae.result()
        mse = mse.result()
        rmse = rmse.result()

        rmse = tf.sqrt(mse).numpy()
        emd = losses.earth_movers_distance(mos, predictions).numpy()

        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"EMD: {emd:.4f}")

        mae = np.abs(predictions - mos)
        print(f"highest error: {np.max(mae)}")

        merged = np.stack((predictions, mos, mae, img_paths), axis=-1)
        indices = np.argsort(merged[:, 2].astype(float))[::-1]
        merged = merged[indices]

        print(f"Top {PRINT_LIMIT} predictions with highest error:")
        for i in range(PRINT_LIMIT):
            print(f"{i+1}. {float(merged[i,0]):.3f} for {float(merged[i,1]):.3f} (error {float(merged[i,2]):.4f}): {merged[i,3]}")

        # histogram
        plt.hist(mos, bins=100)
        plt.hist(predictions, bins=100)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram')
        plt.savefig('histogram.png')

if __name__ == '__main__':
    main()
