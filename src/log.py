import os, datetime, configparser

STATUS_FILE = 'status.ini'
LOG_FILE = 'log.txt'

def status_exists():
    return os.path.exists(STATUS_FILE)

def read_status():
    config = configparser.ConfigParser()
    
    config.read(STATUS_FILE)
    
    epoch = config.getint('progress', 'epoch', fallback=0)
    batch = config.getint('progress', 'batch', fallback=0)
    
    return {'epoch': epoch, 'batch': batch}

def write_status(status):
    config = configparser.ConfigParser()
    
    config['progress'] = {
        'epoch': status['epoch'],
        'batch': status['batch']
    }
    
    with open(STATUS_FILE, 'w') as configfile:
        config.write(configfile)

def log(message):
    with open(LOG_FILE, 'a') as file:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file.write(f"[{timestamp}] {message}\n")

def logprint(message):
    print(message)
    log(message)

def append_csv_history(path, batch, epoch, accuracy, loss):
    isfile = os.path.isfile(path)
    with open(path, 'a') as file:
        if not isfile:
            file.write(f"batch,epoch,accuracy,loss\n")
        file.write(f"{batch},{epoch},{accuracy},{loss}\n")