import os, sys, time, signal, math
import tensorflow as tf
import numpy as np
import tqdm

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{project_dir}/src')

import images, labels, models
from tracker import Tracker

# input
DATA_PATH = f'{project_dir}/data'
MOS_PATH = f'{DATA_PATH}/mos.csv'
IMG_DIRPATH = f'{DATA_PATH}/images/train'

# output
OUTPUT_DIR = f'{project_dir}/output'
MODEL_PATH = f'{OUTPUT_DIR}/model.keras'
BACKUP_PATH = f'{OUTPUT_DIR}/backup.keras'

MAX_HEIGHT = 1440
MAX_WIDTH = 2560
BATCH_SIZE = 5
EPOCHS = 5

tracker = None
mos = None
model = None
img_paths = None
batches_per_epoch = None
total_batches = 0

def signal_handler(sig, frame):
    tracker.logprint(f"Received signal {sig}")
    
    models.save_model(model, BACKUP_PATH)

    tracker.logprint(f"Backup saved at batch {tracker.batch}/{batches_per_epoch} epoch {tracker.epoch}/{EPOCHS}")
    tracker.logprint(f"Exiting...")
    sys.exit(0)

def initialize_resources():
    global mos, img_paths, batches_per_epoch

    data = labels.load_data(MOS_PATH, IMG_DIRPATH)
    img_paths = data[:,0].astype(str)
    mos = data[:,1].astype(np.float32)

    tracker.logprint(f"Detected {len(mos)} labels and {len(img_paths)} images")

    extra_batch_required = len(img_paths) % BATCH_SIZE != 0
    batches_per_epoch = math.floor(len(img_paths)/BATCH_SIZE) + extra_batch_required

def initialize_model():
    try:
        model = models.load_model(MODEL_PATH)
        tracker.logprint(f"Loaded model from file")
    
    except Exception as e:
        model = models.init_model(MAX_HEIGHT, MAX_WIDTH)
        tracker.logprint(f"Initialized new model")
    
    return model

class CustomBatchCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        global total_batches

        tracker.batch = batch + 1
        total_batches += 1
        tracker.log(f"Completed batch {tracker.batch}/{batches_per_epoch} of epoch {tracker.epoch}/{EPOCHS}")

        tracker.save_status()
        tracker.append_csv_history(total_batches, logs['mean_absolute_error'], logs['loss'])
        
        tracker.log(f"Saved status and history")

    def on_epoch_end(self, epoch, logs=None):
        tracker.epoch = epoch + 1
        tracker.batch = 0

        tracker.log(f"Completed epoch {tracker.epoch}/{EPOCHS} completed")
        
        tracker.save_status()
        models.save_model(self.model, MODEL_PATH)
        
        tracker.log(f"Saved model")

def load_img(path, label):
    image = images.load_img(path, MAX_HEIGHT, MAX_WIDTH)
    return image, label

def main():
    global model, tracker

    tracker = Tracker(OUTPUT_DIR)

    tracker.logprint("Program starting up...")
    
    initialize_resources()
    model = initialize_model()

    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mos))
    dataset = dataset.map(load_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    custom_callback = CustomBatchCallback()

    history = model.fit(dataset, verbose=1, initial_epoch=tracker.epoch, epochs=EPOCHS, callbacks=[custom_callback])
    
    tracker.logprint("Program completed")

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    main()
