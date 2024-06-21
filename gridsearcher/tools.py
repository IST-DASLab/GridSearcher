import os
import random
import time
import yaml
import platform
from tqdm import tqdm
from enum import Enum
from .file_locker import lock_acquire, lock_release

class GSExe(Enum):
    PYTHON = 'python3'
    MOSAICML_COMPOSER = 'composer'

class GSKeyValSep(Enum):
    SPACE = ' '
    EQUAL = '='

def validate_constructor_params(
        script: str,
        exe: GSExe = GSExe.PYTHON,
        key_value_separator: GSKeyValSep = GSKeyValSep.SPACE):
    """
        Performs some checks on the constructor parameters.
    """
    # assert os.path.isfile(script), 'Script does not exist'
    assert script.endswith('.py'), 'Script does not end with .py'
    assert isinstance(exe, GSExe), f'Variable exe must be of type {GSExe}'
    assert exe.value in [x.value for x in GSExe], 'exe must be either "python" or "composer"'
    assert isinstance(key_value_separator, GSKeyValSep), f'Variable exe must be of type {GSKeyValSep}'
    assert key_value_separator.value in [x.value for x in GSKeyValSep]

def pause_process(seconds, message=None):
    """
        Pauses the process for specified number of seconds and prints a message before, if specified.
    """
    if message is not None:
        print(message)
    for _ in tqdm(range(seconds)):
        time.sleep(1)

def on_windows():
    """
        Checks whether the operating system is Windows or not
    """
    return platform.system().lower() == 'windows'

def read_yaml(file):
    """
        Reads YAML file and returns a dictionary of the contents.
    """
    with open(file) as f:
        data = yaml.load(f, Loader=yaml.loader.SafeLoader)
        return data

def waiting_worker(params):
    """
        This method will run an experiment with a single element of the cartesian product, on a single process.
    """
    exe, index, cmd, root, cmd_dict, gpu_processes_count, gpus, max_jobs, dist_train, launch_blocking, torchrun = params

    n_gpus = len(gpus)
    """
        Each process sleeps index+5 seconds, where index is the command index. This is necessary because the scripts do not allocate
        GPU memory immediately
    """
    time.sleep(index + 5)

    if not dist_train:
        n_gpus = 1
        while True:
            """
                The shared dictionary gpu_processes_count has key=gpu id and value=number of processes on that GPU
                We sort this dictionary based on values (item[1]) to get least busy GPUs at the first index
            """
            sorted_items = sorted(gpu_processes_count.items(), key=lambda item: item[1]) # this will be a list
            gpu, count = sorted_items[0] # sort ASC by processes count, this is the least busy GPU (count = current number of jobs on GPU)

            # if there are multiple GPUs with minimal number of processes, then pick a random GPU from them
            i = 1
            while i < n_gpus and sorted_items[i][0] == count: # advance i to the first GPU that has a different number of jobs (!= count)
                i += 1
            if count < max_jobs: # if we can fit another job there
                gpu = random.choice([g for g, c in sorted_items[:i]]) # randomly generate a GPU id among the least busy ones)
                lock_acquire() # acquire the lock to change the shared dictionary gpu_processes_count
                gpu_processes_count[gpu] += 1 # increase number of jobs for the GPU
                lock_release()
                break

            print(f'All GPUs in have {max_jobs} jobs, waiting 60 seconds...')
            time.sleep(60)

    # create the root folder, e.g. param_name_for_exp_root_folder
    os.makedirs(root, exist_ok=True)

    # write all parameters to the arguments file
    with open(os.path.join(root, 'arguments.txt'), 'w') as w:
        for k, v in cmd_dict.items():
            if k.startswith('_'):
                w.write(f'{k[1:]}={v}\n')

    # set CUDA_VISIBLE_DEVICES variable
    if dist_train:
        gpus = ",".join(map(str, gpus))
        cvd = f'CUDA_VISIBLE_DEVICES={gpus}'
    else:
        cvd = f'CUDA_VISIBLE_DEVICES={gpu}' # the randomly chosen GPU

    # set CUDA_LAUNCH_BLOCKING variable
    clb = 'CUDA_LAUNCH_BLOCKING=1' if launch_blocking else ''

    if torchrun:
        single_proc_extra_args = '--rdzv-backend=c10d --rdzv-endpoint=localhost:0' if n_gpus == 1 else ''
        cmd = f'{clb} {cvd} torchrun --standalone --nnodes=1 --nproc-per-node={n_gpus} {single_proc_extra_args} {cmd}'.strip()
    else:
        cmd = f'{clb} {cvd} {exe} {cmd}'.strip()

    print(cmd)
    code = os.system(cmd)

    if code == 0:
        # write state.finished file to mark that the experiment was finished
        with open(os.path.join(root, 'state.finished'), 'w'):
            pass

    if not dist_train: # if we are not in distributed settings, decrease the number of processes for the GPU that finished the run
        lock_acquire()
        gpu_processes_count[gpu] -= 1
        lock_release()

# def wait_for_gpus_of_user(gpus, max_jobs=None, timeout_seconds=60):
#     """
#         This method waits `timeout_seconds` for all processes of current user to finish on all GPU cards with IDs in `gpus`.
#     """
#     attempts = 1
#     user = os.getlogin()
#     while True:
#         gpus_stat = gpustat.new_query().gpus # get the status of GPUs
#         processes_used_by_user = 0
#         for i in gpus: # i is the ID of a GPU in CUDA_VISIBLE_DEVICES
#             for proc in gpus_stat[i].processes: # proc is a dict containing keys (username, command, gpu_memory_usage, pid)
#                 if proc['username'] == user:
#                     processes_used_by_user += 1
#         if max_jobs is None: # run when GPUs are free
#             if processes_used_by_user == 0: # the script can run now
#                 return
#         else: # run when there are less than max_jobs
#             if processes_used_by_user < max_jobs:
#                 return
#
#         # block the script here
#         print(f'(#{attempts}) {user} has processes running on at least one GPU from {gpus}, waiting {timeout_seconds} seconds...')
#         attempts += 1
#         time.sleep(timeout_seconds)

# def get_free_gpu(gpus, max_jobs, attempts=0):
#     """
#     Returns the first GPU from `gpus` that has less than `max_jobs` running for the current user
#     """
#     user = os.getlogin()
#     can_run_on_gpu = [False] * len(gpus) # flags telling whether we can run the script on a gpu in `gpus`
#     gpu_proc_count = [0] * len(gpus)
#
#     gpu_stat = gpustat.new_query().gpus
#     for i, gpu_id in enumerate(gpus):
#         user_processes = [p for p in gpu_stat[gpu_id].processes if p['username'] == user]
#         gpu_proc_count[i] = len(user_processes)
#         # can_run_on_gpu[i] = (len(user_processes) < max_jobs)
#
#     least_busy_gpu_count = None
#     least_busy_gpu_index = None
#     for i, count in enumerate(gpu_proc_count):
#         if count < max_jobs:
#             if least_busy_gpu_count is None or count < least_busy_gpu_count:
#                 least_busy_gpu_count = count
#                 least_busy_gpu_index = gpus[i]
#
#     if least_busy_gpu_count is not None:
#         return least_busy_gpu_index
#
#     # # if one flag is True, then pick the gpu from that index and run on it
#     # available_gpus = [i for i, flag in enumerate(can_run_on_gpu) if flag]
#     # if len(available_gpus) > 0:
#     #     gpu = random.choice(available_gpus)
#     #     return gpu
#
#     # wait 60 seconds then try again
#     print(f'All GPUs in {gpus} have {max_jobs} jobs, waiting 60 seconds...')
#     time.sleep(60)
#     get_free_gpu(gpus, max_jobs, attempts + 1)

# def wait_for_processes(pids, timeout_seconds=60):
#     if pids is not None:
#         attempts = 1
#         while sum([psutil.pid_exists(pid) for pid in pids]) > 0:
#             print(f'(#{attempts}) at least one process from {pids} is still running, waiting {timeout_seconds} seconds...')
#             attempts += 1
#             time.sleep(timeout_seconds)
