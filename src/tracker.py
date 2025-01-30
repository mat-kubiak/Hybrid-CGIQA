import os, datetime, configparser

def _get_last_line(path):
    with open(path, "rb") as f:
        f.seek(-2, 2)
        while f.read(1) != b"\n":
            f.seek(-2, 1)
        last_line = f.readline().decode()
    return last_line

class Tracker:

    def __init__(self, output_dir):
        self.log_path = f'{output_dir}/log.txt'
        self.status_path = f'{output_dir}/status.ini'
        self.history_path = f'{output_dir}/batch-history.csv'

        self.batch = 0
        self.epoch = 0
        self.batch_counter = 0

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if not os.path.isfile(self.status_path):
            self.save_status()
            self.logprint("Created status file")
        else:
            self.load_status()
            obj = {'epoch': self.epoch, 'batch': self.batch}
            self.logprint(f"Loaded status file: {obj}")

    def load_status(self):
        config = configparser.ConfigParser()
        config.read(self.status_path)
        
        self.epoch = config.getint('progress', 'epoch', fallback=0)
        self.batch = config.getint('progress', 'batch', fallback=0) 

    def save_status(self):
        config = configparser.ConfigParser()
        config['progress'] = {
            'epoch': self.epoch,
            'batch': self.batch
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

    def append_csv_history(self, stats):
        keys = stats.keys()
        values = list(map(str, stats.values()))

        isfile = os.path.isfile(self.history_path)
        with open(self.history_path, 'a') as file:
            if not isfile:
                file.write('batch,' + ','.join(keys) + '\n')
            file.write(str(self.batch_counter) + ',' + ','.join(values) + '\n')
        
        self.batch_counter += 1
