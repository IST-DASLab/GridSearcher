import psutil
import gpustat
import argparse
import time
from tqdm import tqdm

def wait_for_pids(pids):
    if not pids:
        return
    while any([psutil.pid_exists(int(pid)) for pid in pids]):
        print(f'waiting for processes {pids} to end...')
        time.sleep(60)

def get_active_gpu_pids():
    gpus = gpustat.new_query().gpus
    num_gpus = len(gpus)
    pids = [0] * num_gpus
    for gid in range(num_gpus):
        for p in gpus[gid]['processes']:
            pids[gid] = p['pid']
    return pids

def get_wait_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait_pids', nargs='+', default=None, required=False)
    parser.add_argument('--wait_secs', type=int, default=None, required=False)
    parser.add_argument('--wait_current', action="store_true", default=False)
    return parser.parse_args()

def wait_for_resources():
    args = get_wait_args()
    if args.wait_current:  # if this option is set, then we will wait for all GPUs to be free (meaning no more processes running on GPUs)
        wait_for_pids(pids=get_active_gpu_pids())
    else:
        if args.wait_pids is not None:  # alternatively, we can wait for explicit PIDs (it can also be the PID of an already running GridSearcher script)
            wait_for_pids(pids=args.wait_pids)

        if args.wait_secs is not None:  # alternatively, we can wait a specific amount of seconds before we start the script
            print(f'Waiting {args.wait_secs} seconds')
            for _ in tqdm(range(args.wait_secs)):
                time.sleep(1)