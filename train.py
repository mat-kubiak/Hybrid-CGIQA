import os, sys, time, signal
import numpy as np

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

status = None
mos = None
model = None

def signal_handler(sig, frame):
    log.logprint(f"Received signal {sig}, exiting...")    
    sys.exit(0)

def main():
    global status, mos, model
    log.logprint("Program starting up...")

    mos = labels.load_labels(MOS_PATH, IMG_DIRPATH)
    log.logprint(f"Loaded {mos.shape[0]} labels")
    if (mos.shape[0] == 0):
        los.logprint("Fatal error: no labels found")
        sys.exit(1)

    if not log.status_exists():
        log.logprint("Created status file")
        log.write_status({'epoch': 0, 'batch': 0})
    
    status = log.read_status()
    log.logprint(f"Loaded status file: {status}")

    if not models.model_exists(MODEL_PATH):
        model = models.init_model(MAX_HEIGHT, MAX_WIDTH)
        log.logprint(f"Initialized new model with max image dims: {MAX_WIDTH}x{MAX_HEIGHT}")
    else:
        try:
            model = models.load_model(MODEL_PATH)
            log.logprint(f"Loaded model from file")
        except Exception as e:
            log.logprint(f"Fatal Error: Could not load model file: {e}")
            sys.exit(1)
    
    # TODO: load image batch and train
    
    log.logprint("Program completed")

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    main()
