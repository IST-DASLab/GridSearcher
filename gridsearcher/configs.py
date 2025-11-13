from dataclasses import dataclass
from types import List, Dict
import ipaddress

def is_valid_ip(address: str) -> bool:
    if address.lower() == "localhost":
        return True
    try:
        ipaddress.ip_address(address)
        return True
    except ValueError:
        return False

@dataclass
class SchedulingConfig:
    """
    Description:
        Represents a scheduling configuration for GridSearcher.
    Attributes:
        distributed_training (bool): whether the experiment uses DataParallel or not. If set to True,
                                     then all GPUs in the `gpus` list will be used for CUDA_VISIBLE_DEVICES.
                                     Otherwise, only one GPU id will be used for CUDA_VISIBLE_DEVICES.
        max_jobs_per_gpu (int): specifies how many processes should run on each GPU at most (num_processes = len(gpus) * max_jobs_per_gpu)
        gpus (List[int]): a list containing IDs of GPUs you want to run your tasks on
        params_values (Dict[str, List]): a dictionary that contains the grid for your hyper-parameters (the cartesian product will be computed)
    """
    distributed_training: bool
    max_jobs_per_gpu: int
    gpus: List[int]
    params_values: Dict[str, List]

    def __post_init__(self):
        assert type(self.distributed_training) is bool
        assert type(self.max_jobs_per_gpu) is bool
        assert type(self.gpus) is list and all([type(gpu) is int for gpu in self.gpus])
        assert type(self.params_values) is dict
        assert all([type(k) is str and type(v) is list for k, v in self.params_values.items()])

        # remove duplicates
        for k, v in self.params_values.items():
            self.params_values[k] = list(set(v))

@dataclass
class TorchRunConfig:
    """
    Description:
        Represents a configuration for torchrun for GridSearcher.
    Attributes:
        launch_blocking (int): if set to 1, then add CUDA_LAUNCH_BLOCKING=1 to the command line
        torchrun (bool): whether to run with torchrun or not
        master_addr (str):
        master_port (int):
        rdzv_backend (str):
    """
    launch_blocking: int = 0
    torchrun: bool = True
    master_addr: str = '127.0.0.1'
    master_port: int = 29500
    rdzv_backend: str = 'c10d'

    def __post_init__(self):
        assert self.launch_blocking in [0, 1]
        assert type(self.torchrun) is bool
        assert is_valid_ip(self.master_addr)
        assert type(self.master_port) is int
        assert self.rdzv_backend in ['c10d', 'static']