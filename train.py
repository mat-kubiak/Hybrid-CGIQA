import os, sys, time, signal, math, datetime, random, traceback
import tensorflow as tf
import numpy as np
from tensorboard.plugins.hparams import api as hp

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{PROJECT_DIR}/src')

import images, labels, models
from tracker import Tracker
from batchcallback import BatchCallback
from weights_histogram_callback import WeightsHistogramCallback

# input
DATA_DIR = f'{PROJECT_DIR}/data'
MOS_FILE = f'{DATA_DIR}/mos.csv'
FIT_IMG_DIR = f'{DATA_DIR}/images/train'
VAL_IMG_DIR = f'{DATA_DIR}/images/test'

# logging
TIMESTAMP = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
LOG_FILE = f'{PROJECT_DIR}/logs/{TIMESTAMP}.txt'

# output
MODEL_NAME = ''
OUTPUT_DIR = f'{PROJECT_DIR}/output/{MODEL_NAME}'

STATUS_FILE = f'{OUTPUT_DIR}/status.ini'
MODEL_FILE = f'{OUTPUT_DIR}/model.keras'
BACKUP_FILE = f'{OUTPUT_DIR}/backup.keras'

HEIGHT = 256
WIDTH = 256
MODEL_HEIGHT = 224
MODEL_WIDTH = 224

FIT_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
EPOCHS = 50
FIXED_VAL_IMG_DIMS = True
IS_CATEGORICAL = False
GAUSSIAN_NOISE = 0

# if set, limits data to n first samples
FIT_LIMIT = None
VAL_LIMIT = None

tracker = None
model = None
fit_mos = None
fit_imgs = None
val_mos = None
val_imgs = None
batches_per_epoch = None

SEED = 23478
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

tf.keras.config.enable_unsafe_deserialization()

def signal_handler(sig, frame):
    tracker.logprint(f"Received signal {sig}")
    
    models.save_model(model, BACKUP_FILE)

    tracker.logprint(f"Backup saved at batch {tracker.batch}/{batches_per_epoch} epoch {tracker.epoch}/{EPOCHS}")
    tracker.logprint(f"Exiting...")
    sys.exit(0)

def initialize_resources():
    global fit_mos, fit_imgs, val_mos, val_imgs, batches_per_epoch

    fit_imgs = images.get_image_list(FIT_IMG_DIR)
    fit_mos = labels.load(MOS_FILE, FIT_IMG_DIR, IS_CATEGORICAL)
    tracker.logprint(f"Detected {len(fit_mos)} labels and {len(fit_imgs)} images")

    if FIT_LIMIT != None:
        fit_imgs = fit_imgs[:FIT_LIMIT]
        fit_mos = fit_mos[:FIT_LIMIT]
        tracker.logprint(f"Limiting data to {FIT_LIMIT} first samples")

    extra_batch_required = len(fit_imgs) % FIT_BATCH_SIZE != 0
    batches_per_epoch = math.floor(len(fit_imgs)/FIT_BATCH_SIZE) + extra_batch_required

    val_mos = labels.load(MOS_FILE, VAL_IMG_DIR, IS_CATEGORICAL)
    val_imgs = images.get_image_list(VAL_IMG_DIR)
    tracker.logprint(f"Detected {len(val_mos)} validation labels and validation {len(val_imgs)} images")

    if VAL_LIMIT != None:
        val_mos = val_mos[:FIT_LIMIT]
        val_imgs = val_imgs[:FIT_LIMIT]
        tracker.logprint(f"Limiting validation data to {FIT_LIMIT} first samples")

def log_hparams():
    global model
    hparams = {
        'resolution': f'{HEIGHT}x{WIDTH}',
        'batch_size': FIT_BATCH_SIZE,
        'epochs': EPOCHS,
        'output': 'categorical' if IS_CATEGORICAL else 'numerical',
        'total_layers': len(model.layers),
        'optimizer': model.optimizer.__class__.__name__,
        'trainable_params': model.count_params(),
        'loss': model.loss.__class__.__name__,
        'label-noise': GAUSSIAN_NOISE,
    }

    print(hparams)

    writer = tf.summary.create_file_writer(OUTPUT_DIR)
    with writer.as_default():
        hp.hparams(hparams)

def initialize_model():
    global model
    
    if os.path.isfile(MODEL_FILE):
        try:
            model = models.load_model(MODEL_FILE)
            tracker.logprint(f"Loaded model from file")
        except Exception as e:
            tracker.logprint(f"Fatal error while loading model file at: {MODEL_FILE}")
            traceback.print_exc()
            sys.exit(-1)
    else:
        try:
            model = models.init_model(None, None, IS_CATEGORICAL, gaussian=GAUSSIAN_NOISE)
            tf.keras.utils.plot_model(model, to_file=f"{OUTPUT_DIR}/arch.png", show_shapes=True, show_dtype=True, show_layer_names=True)
            tracker.logprint(f"Initialized new model")
        except Exception as e:
            tracker.logprint(f"Fatal error while initializing model")
            traceback.print_exc()
            sys.exit(-1)
    
    model.summary()
    log_hparams()

def load_val_image(path, label):
    image = images.load_image(path, MODEL_HEIGHT, MODEL_WIDTH)
    return image, label

def load_fit_image(path, label):
    image = images.load_image(path, HEIGHT, WIDTH)
    image = images.random_crop_image(image, MODEL_HEIGHT, MODEL_WIDTH)
    return image, label

def main():
    global model, tracker

    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)

    tracker = Tracker(log_path=LOG_FILE, status_path=STATUS_FILE)

    tracker.logprint("Program starting up...")

    initialize_resources()
    initialize_model()

    dataset = (tf.data.Dataset.from_tensor_slices((fit_imgs, fit_mos))
        .shuffle(buffer_size=1000)
        .map(load_fit_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(FIT_BATCH_SIZE)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    if FIXED_VAL_IMG_DIMS:
        val_dataset = (tf.data.Dataset.from_tensor_slices((val_imgs, val_mos))
            .map(load_val_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(VAL_BATCH_SIZE)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
    else:
        val_dataset = (tf.data.Dataset.from_tensor_slices((val_imgs, val_mos))
            .map(load_val_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .padded_batch(VAL_BATCH_SIZE,
                padded_shapes=([None, None, 3], []), 
                padding_values=(0.0, 0.0))
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    batch_callback = BatchCallback(tracker, EPOCHS, MODEL_FILE, batches_per_epoch)
    weights_callback = WeightsHistogramCallback(log_dir=OUTPUT_DIR)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=OUTPUT_DIR,
        write_graph=True,
        histogram_freq=1,
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        dataset,
        verbose=1,
        validation_data=val_dataset,
        initial_epoch=tracker.epoch,
        epochs=EPOCHS,
        callbacks=[batch_callback, tensorboard_callback, weights_callback, early_stopping]
    )

    tracker.logprint("Program completed")

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    main()
