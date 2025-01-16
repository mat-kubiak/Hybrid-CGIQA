import os, sys, time, signal
import numpy as np

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{project_dir}/src')

import images, labels, log, models, props

MOS_PATH = f'{project_dir}/data/raw/mos.csv'

status = None
mos = None
model = None

def signal_handler(sig, frame):
    log.logprint(f"Received signal {sig}, exiting...")    
    sys.exit(0)

def main():
    global status, mos, model
    log.logprint("Program starting up...")

    mos = labels.load_labels(MOS_PATH)
    log.logprint(f"Loaded {mos.shape[0]} labels")
    if (mos.shape[0] == 0):
        los.logprint("Fatal error: no labels found")
        sys.exit(1)

    if not log.status_exists():
        log.logprint("Created status file")
        log.write_status({'epoch': 0, 'batch': 0})
    
    status = log.read_status()
    log.logprint(f"Loaded status file: {status}")

    # TODO: load model and train

    log.logprint("Program completed")

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    main()