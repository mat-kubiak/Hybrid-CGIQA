import os, sys, time, signal

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{project_dir}/src')

import log

def signal_handler(sig, frame):
    log.logprint(f"Received signal {sig}, preparing to exit...")
    sys.exit(0)

def main():
    log.logprint("Program started, it will now count to 1000")

    for i in range(1000):
        log.logprint(i)
        time.sleep(1)

    log.logprint("Program ended counting without interruption")

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    main()