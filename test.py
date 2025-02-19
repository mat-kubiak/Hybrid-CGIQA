import os, sys, traceback
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, kendalltau, wasserstein_distance

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{PROJECT_DIR}/src')

import images, labels, models

MODEL_NAME = ''
OUTPUT_DIR = f'{PROJECT_DIR}/output/{MODEL_NAME}'
MODEL_PATH = f'{OUTPUT_DIR}/model.keras'

# can be set to 'movie' or 'game' to enforce testing on only one type of image
IMAGE_TYPE = ''

floor = '_' if IMAGE_TYPE != '' else ''
RESULTS_FILE = f'{OUTPUT_DIR}/predictions{floor}{IMAGE_TYPE}.npy'
HISTOGRAM_FILE = f'{OUTPUT_DIR}/histogram{floor}{IMAGE_TYPE}.png'

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

def filter_data(img_paths, mos):
    img_names = [os.path.basename(path) for path in img_paths]

    f_paths = []
    f_mos = []
    for i in range(len(img_names)):
        if IMAGE_TYPE in img_names[i]:
            f_paths.append(img_paths[i])
            f_mos.append(mos[i])

    return (f_paths, f_mos)

def main():
    global HEIGHT, WIDTH, IS_CATEGORICAL

    img_paths = images.get_image_list(IMG_DIRPATH)
    mos = labels.load_continuous(MOS_PATH, IMG_DIRPATH)
    print(f"Detected {len(mos)} labels and {len(img_paths)} images")

    if LIMIT != None and LIMIT < len(mos):
        print(f"Limited images to {LIMIT}")
        mos = mos[:LIMIT]
        img_paths = img_paths[:LIMIT]

    if IMAGE_TYPE != '':
        img_paths, mos = filter_data(img_paths, mos)
        print(f"Limiting image type to '{IMAGE_TYPE}': detected {len(mos)} images")

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

    if model.output_shape[-1] == 41:
        print(f"Testing for classification models not included, sorry!")
        exit()

    dataset = (tf.data.Dataset.from_tensor_slices((img_paths, mos))
        .map(load_image)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    predictions = model.predict(dataset).flatten()
    np.save(RESULTS_FILE, predictions)

    params, covariance = curve_fit(logistic_function, predictions, mos, p0=[1, 1, 1, 1, 1])
    beta1, beta2, beta3, beta4, beta5 = params

    print("Fitted Parameters:")
    print(f"β1 = {beta1}, β2 = {beta2}, β3 = {beta3}, β4 = {beta4}, β5 = {beta5}")

    predictions = logistic_function(predictions, *params)

    tf_metrics = [
        ['MAE', tf.keras.metrics.MeanAbsoluteError()],
        ['MSE', tf.keras.metrics.MeanSquaredError()],
        ['RMSE', tf.keras.metrics.RootMeanSquaredError()]
    ]
    for metric in tf_metrics:
        metric[1].update_state(mos, predictions)
        print(f'{metric[0]}: {metric[1].result():.4f}')

    print(f"EMD: {wasserstein_distance(mos, predictions):.4f}")
    print(f'PLCC: {pearsonr(mos, predictions)[0]:.4f}')
    print(f'SRCC: {spearmanr(mos, predictions)[0]:.4f}')
    print(f'KRCC: {kendalltau(mos, predictions)[0]:.4f}')

    mae = np.abs(predictions - mos)
    print(f"highest error: {np.max(mae)}")

    merged = np.stack((predictions, mos, mae, img_paths), axis=-1)
    indices = np.argsort(merged[:, 2].astype(float))[::-1]
    merged = merged[indices]

    print(f"Top {PRINT_LIMIT} predictions with highest error:")
    for i in range(PRINT_LIMIT):
        print(f"{i+1}. {float(merged[i,0]):.3f} for {float(merged[i,1]):.3f} (error {float(merged[i,2]):.4f}): {merged[i,3]}")

    # histogram
    bins = 30
    plt.hist(mos, bins=bins, histtype='step', color='red', label='True')
    plt.hist(predictions, bins=bins, histtype='step', color='blue', label='Predicted')
    plt.xlabel('MOS Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(HISTOGRAM_FILE, dpi=300)

if __name__ == '__main__':
    main()
