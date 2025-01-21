import os, sys, time, signal, math
import numpy as np
import tqdm

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{project_dir}/src')

import images, labels, log, models

# path to the database containing the images and mos.csv
# change according to your needs
DATA_PATH = f'{project_dir}/data'

MODEL_PATH = f'{project_dir}/model.keras'
MOS_PATH = f'{DATA_PATH}/mos.csv'
IMG_DIRPATH = f'{DATA_PATH}/images/train'

MAX_HEIGHT = 1080
MAX_WIDTH = 1920
RATINGS = 41 # range 1.0, 5.0 with step 0.1
BATCH_SIZE = 5
BATCHES = None
EPOCHS = 10

# Debug Mode
# put to true to simulate logic without actually training
DEBUG = False

status = None
mos = None
model = None

def signal_handler(sig, frame):
    log.logprint(f"Received signal {sig}, exiting...")    
    sys.exit(0)

def main():
    global status, mos, model, BATCHES
    log.logprint("Program starting up...")
    if DEBUG:
        log.logprint("Started with DEBUG=True")

    mos = labels.load_labels(MOS_PATH, IMG_DIRPATH)
    log.logprint(f"Loaded {mos.shape[0]} labels")
    if (mos.shape[0] == 0):
        log.logprint("Fatal error: no labels found")
        sys.exit(1)

    if not log.status_exists():
        log.logprint("Created status file")
        log.write_status({'epoch': 0, 'batch': 0})
    
    status = log.read_status()
    log.logprint(f"Loaded status file: {status}")

    if (status['epoch'] >= EPOCHS):
        log.logprint(f"Target number of epochs {EPOCHS} already achieved. Exiting...")
        sys.exit(0)

    if not models.model_exists(MODEL_PATH):
        model = models.init_model(MAX_HEIGHT, MAX_WIDTH, RATINGS)
        log.logprint(f"Initialized new model with max image dims: {MAX_WIDTH}x{MAX_HEIGHT}")
    else:
        try:
            model = models.load_model(MODEL_PATH)
            log.logprint(f"Loaded model from file")
        except Exception as e:
            log.logprint(f"Fatal Error: Could not load model file: {e}")
            sys.exit(1)
    
    # load image list
    img_list = images.get_image_list(IMG_DIRPATH)
    log.logprint(f"Found {len(img_list)} images")
    if len(img_list) % BATCH_SIZE != 0:
        log.logprint("Warning: number of images is not divisible by batch size")
    BATCHES = math.floor(len(img_list)/BATCH_SIZE)

    running = True
    while(running):
        
        start_i = status['batch'] * BATCH_SIZE
        end_i = (status['batch'] + 1) * BATCH_SIZE

        # prepare images
        x_train = np.zeros((BATCH_SIZE, MAX_HEIGHT, MAX_WIDTH, 3))
        for i in tqdm.tqdm(range(0, BATCH_SIZE)):
            if DEBUG:
                continue
            img_path = f"{IMG_DIRPATH}/{img_list[start_i + i]}"
            img = images.load_img(img_path, MAX_HEIGHT, MAX_WIDTH)
            
            x_train[i, :, :, :] = img
        
        # prepare labels
        y_train = mos[start_i:end_i, :]

        # train
        if not DEBUG:
            model = models.train_model(model, x_train, y_train, 1, BATCH_SIZE)
        
        status['batch'] += 1
        if (status['batch'] >= BATCHES):
            status['epoch'] += 1
            status['batch'] = 0
            log.logprint(f"training for epoch {status['epoch']}/{EPOCHS} completed")
        
        log.logprint(f"training batch {status['batch']}/{BATCHES} of epoch {status['epoch']}/{EPOCHS} completed")

        if (status['epoch'] >= EPOCHS):
            log.logprint(f"Achieved a goal of {EPOCHS} epochs, training is completed")
            running = False

        log.logprint(f"Saving status and model...")
        log.write_status(status)
        if not DEBUG:
            models.save_model(model, MODEL_PATH)
        log.logprint(f"Saved status and model")
    
    log.logprint("Program completed")

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    main()
