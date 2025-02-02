import os, datetime, configparser

class Tracker:

    def __init__(self, log_path, status_path):
        self.log_path = log_path
        self.status_path = status_path

        self.batch = 0
        self.epoch = 0

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(os.path.dirname(status_path), exist_ok=True)
        
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
