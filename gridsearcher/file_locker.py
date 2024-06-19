import os

LOCK_FILE = 'locker.lock'

# https://github.com/dmfrey/FileLock/blob/master/filelock/filelock.py
# https://superfastpython.com/multiprocessing-pool-mutex-lock/

def lock_acquire():
    while True:
        if not os.path.isfile(LOCK_FILE):
            break
    open(LOCK_FILE, 'w').close()

def lock_release():
    if os.path.isfile(LOCK_FILE):
        try:
            os.remove(LOCK_FILE)
        except:
            pass
