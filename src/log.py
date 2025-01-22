import os, datetime, configparser

def status_exists(path):
    return os.path.exists(path)

def read_status(path):
    config = configparser.ConfigParser()
    
    config.read(path)
    
    epoch = config.getint('progress', 'epoch', fallback=0)
    batch = config.getint('progress', 'batch', fallback=0)
    
    return {'epoch': epoch, 'batch': batch}

def write_status(path, status):
    config = configparser.ConfigParser()
    
    config['progress'] = {
        'epoch': status['epoch'],
        'batch': status['batch']
    }
    
    with open(path, 'w') as configfile:
        config.write(configfile)

def log(path, message):
    with open(path, 'a') as file:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file.write(f"[{timestamp}] {message}\n")

def logprint(path, message):
    print(message)
    log(path, message)

def append_csv_history(path, batch, mse, loss):
    isfile = os.path.isfile(path)
    with open(path, 'a') as file:
        if not isfile:
            file.write(f"batch,mse,loss\n")
        file.write(f"{batch},{mse},{loss}\n")