import os, sys, time, signal, math
import tensorflow as tf
import numpy as np
import tqdm

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{project_dir}/src')

import images, labels, tracker, models

# path to the database containing the images and mos.csv
# change according to your needs
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
BATCHES = None
EPOCHS = 5

tracker = None
status = None
mos = None
model = None
img_paths = None
total_batches = 0

def signal_handler(sig, frame):
    global status
    tracker.logprint(f"Received signal {sig}")
    
    models.save_model(model, BACKUP_PATH)

    tracker.logprint(f"Backup saved at batch {status['batch']}/{BATCHES} epoch {status['epoch']}/{EPOCHS}")
    tracker.logprint(f"Exiting...")
    sys.exit(0)

def initialize_resources():
    global status, mos, img_paths, BATCHES

    # load labels
    mos = labels.load_labels(MOS_PATH, IMG_DIRPATH)
    tracker.logprint(f"Loaded {mos.shape[0]} labels")
    
    if (mos.shape[0] == 0):
        tracker.logprint("Fatal error: no labels found")
        sys.exit(1)

    # image list
    img_paths = images.get_image_list(IMG_DIRPATH)
    img_paths = IMG_DIRPATH + "/" + img_paths
    tracker.logprint(f"Found {len(img_paths)} images")

    # batches
    if len(img_paths) % BATCH_SIZE != 0:
        tracker.logprint("Warning: number of images is not divisible by batch size")
    BATCHES = math.floor(len(img_paths)/BATCH_SIZE)

    # status
    if not tracker.status_exists():
        tracker.logprint("Created status file")
        tracker.write_status({'epoch': 0, 'batch': 0})
    
    status = tracker.read_status()
    tracker.logprint(f"Loaded status file: {status}")

def initialize_model():
    global model

    if not models.model_exists(MODEL_PATH):
        model = models.init_model(MAX_HEIGHT, MAX_WIDTH)
        tracker.logprint(f"Initialized new model with max image dims: {MAX_WIDTH}x{MAX_HEIGHT}")
        return
    
    try:
        model = models.load_model(MODEL_PATH)
        tracker.logprint(f"Loaded model from file")
    except Exception as e:
        tracker.logprint(f"Fatal Error: Could not load model file: {e}")
        sys.exit(1)

class CustomBatchCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        global status, total_batches

        status['batch'] = batch + 1
        total_batches += 1
        tracker.log(f"Completed batch {status['batch']}/{BATCHES} of epoch {status['epoch']}/{EPOCHS}")

        tracker.write_status(status)
        tracker.append_csv_history(total_batches, logs['mean_absolute_error'], logs['loss'])
        
        tracker.log(f"Saved status and history")

    def on_epoch_end(self, epoch, logs=None):
        global status

        status['epoch'] = epoch + 1
        status['batch'] = 0

        tracker.log(f"Completed epoch {status['epoch']}/{EPOCHS} completed")
        
        tracker.write_status(status)
        models.save_model(self.model, MODEL_PATH)
        
        tracker.log(f"Saved model")

def main():
    global status, model, tracker

    tracker = Tracker(OUTPUT_DIR)

    tracker.logprint("Program starting up...")
    
    initialize_resources()
    initialize_model()

    train_dataset = tf.data.Dataset.from_tensor_slices((img_paths, mos))
    train_dataset = train_dataset.map(lambda path, label: (images.load_img(path, MAX_HEIGHT, MAX_WIDTH), label), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    custom_callback = CustomBatchCallback()

    history = model.fit(train_dataset, verbose=1, initial_epoch=status['epoch'], epochs=EPOCHS, callbacks=[custom_callback])
    
    tracker.logprint("Program completed")

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    main()
