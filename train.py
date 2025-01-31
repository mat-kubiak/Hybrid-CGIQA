import os, sys, time, signal, math, datetime
import tensorflow as tf
import numpy as np

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{project_dir}/src')

import images, labels, models
from tracker import Tracker
from batchcallback import BatchCallback

# input
DATA_DIR = f'{project_dir}/data'
MOS_PATH = f'{DATA_DIR}/mos.csv'
FIT_IMG_DIR = f'{DATA_DIR}/images/train'
VAL_IMG_DIR = f'{DATA_DIR}/images/test'

# output
OUTPUT_DIR = f'{project_dir}/output'
MODEL_PATH = f'{OUTPUT_DIR}/model.keras'
BACKUP_PATH = f'{OUTPUT_DIR}/backup.keras'

HEIGHT = 512
WIDTH = 512
FIT_BATCH_SIZE = 20
VAL_BATCH_SIZE = 20
EPOCHS = 50
IS_CATEGORICAL = False

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

def signal_handler(sig, frame):
    tracker.logprint(f"Received signal {sig}")
    
    models.save_model(model, BACKUP_PATH)

    tracker.logprint(f"Backup saved at batch {tracker.batch}/{batches_per_epoch} epoch {tracker.epoch}/{EPOCHS}")
    tracker.logprint(f"Exiting...")
    sys.exit(0)

def initialize_resources():
    global fit_mos, fit_imgs, val_mos, val_imgs, batches_per_epoch

    fit_imgs = images.get_image_list(FIT_IMG_DIR)
    fit_mos = labels.load(MOS_PATH, FIT_IMG_DIR, IS_CATEGORICAL)
    tracker.logprint(f"Detected {len(fit_mos)} labels and {len(fit_imgs)} images")

    if FIT_LIMIT != None:
        fit_imgs = fit_imgs[:FIT_LIMIT]
        fit_mos = fit_mos[:FIT_LIMIT]
        tracker.logprint(f"Limiting data to {FIT_LIMIT} first samples")

    extra_batch_required = len(fit_imgs) % FIT_BATCH_SIZE != 0
    batches_per_epoch = math.floor(len(fit_imgs)/FIT_BATCH_SIZE) + extra_batch_required

    val_mos = labels.load(MOS_PATH, VAL_IMG_DIR, IS_CATEGORICAL)
    val_imgs = images.get_image_list(VAL_IMG_DIR)
    tracker.logprint(f"Detected {len(val_mos)} validation labels and validation {len(val_imgs)} images")

    if VAL_LIMIT != None:
        val_mos = val_mos[:FIT_LIMIT]
        val_imgs = val_imgs[:FIT_LIMIT]
        tracker.logprint(f"Limiting validation data to {FIT_LIMIT} first samples")

def initialize_model():
    try:
        model = models.load_model(MODEL_PATH)
        tracker.logprint(f"Loaded model from file")
    
    except Exception as e:
        model = models.init_model(HEIGHT, WIDTH, IS_CATEGORICAL)
        tf.keras.utils.plot_model(model, to_file=f"{OUTPUT_DIR}/arch.png", show_shapes=True, show_dtype=True, show_layer_names=True, show_trainable=True)
        tracker.logprint(f"Initialized new model")
    
    model.summary()
    return model

def load_image(path, label):
    image = images.load_image(path, HEIGHT, WIDTH)
    return image, label

def main():
    global model, tracker

    tracker = Tracker(OUTPUT_DIR)

    tracker.logprint("Program starting up...")
    
    initialize_resources()
    model = initialize_model()

    augment_image = models.get_augmentation_model()

    dataset = (tf.data.Dataset.from_tensor_slices((fit_imgs, fit_mos))
        .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=1000)
        .batch(FIT_BATCH_SIZE)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    val_dataset = (tf.data.Dataset.from_tensor_slices((val_imgs, val_mos))
        .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(VAL_BATCH_SIZE)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    batch_callback = BatchCallback(tracker, EPOCHS, MODEL_PATH, batches_per_epoch)
    csv_logger = tf.keras.callbacks.CSVLogger(f"{OUTPUT_DIR}/epoch-history.csv", append=True)

    log_dir = f"{OUTPUT_DIR}/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(
        dataset,
        verbose=1,
        validation_data=val_dataset,
        initial_epoch=tracker.epoch,
        epochs=EPOCHS,
        callbacks=[batch_callback, csv_logger, tensorboard_callback]
    )

    tracker.logprint("Program completed")

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    main()
