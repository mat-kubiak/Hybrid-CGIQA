import os, datetime, configparser

class Tracker:

    def __init__(self, output_dir):
        self.log_path = f'{output_dir}/log.txt'
        self.status_path = f'{output_dir}/status.ini'
        self.history_path = f'{output_dir}/batch-history.csv'

        self.batch = 0
        self.epoch = 0

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if not os.path.isfile(self.status_path):
            self.save_status()
            self.logprint("Created status file")
        else:
            self.load_status()
            obj = {'epoch': self.epoch}
            self.logprint(f"Loaded status file: {obj}")

    def load_status(self):
        config = configparser.ConfigParser()
        config.read(self.status_path)
        
        self.epoch = config.getint('progress', 'epoch', fallback=0)

    def save_status(self):
        config = configparser.ConfigParser()
        config['progress'] = {
            'epoch': self.epoch,
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
