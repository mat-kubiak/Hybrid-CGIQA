import os, datetime, configparser

class Tracker:

    def __init__(self, output_dir):
        self.log_path = f'{output_dir}/log.txt'
        self.status_path = f'{output_dir}/status.ini'
        self.history_path = f'{output_dir}/history.csv'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def status_exists(self):
        return os.path.exists(self.status_path)

    def read_status(self):
        config = configparser.ConfigParser()
        
        config.read(self.status_path)
        
        epoch = config.getint('progress', 'epoch', fallback=0)
        batch = config.getint('progress', 'batch', fallback=0)
        
        return {'epoch': epoch, 'batch': batch}

    def write_status(self, status):
        config = configparser.ConfigParser()
        
        config['progress'] = {
            'epoch': status['epoch'],
            'batch': status['batch']
        }
        
        with open(self.status_path, 'w') as configfile:
            config.write(configfile)

    def log(self, message):
        with open(self.log_path, 'a') as file:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            file.write(f"[{timestamp}] {message}\n")

    def logprint(self, message):
        print(message)
        self.log(message)

    def append_csv_history(self, batch, mse, loss):
        isfile = os.path.isfile(self.history_path)
        with open(self.history_path, 'a') as file:
            if not isfile:
                file.write(f"batch,mse,loss\n")
            file.write(f"{batch},{mse},{loss}\n")