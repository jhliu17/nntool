from ..accelerator.utils import nvidia_smi_gpu_memory_stats_str as nvidia_smi_gpu_memory_stats_str
from .args import SlurmArgs as SlurmArgs
from _typeshed import Incomplete
from dataclasses import dataclass

class Task:
    argv: Incomplete
    slurm_args: Incomplete
    verbose: Incomplete
    def __init__(self, argv: list[str], slurm_args: SlurmArgs, verbose: bool = False) -> None: ...
    def log(self, msg: str): ...
    def command(self) -> str: ...
    def checkpoint(self): ...

@dataclass
class DistributedArgs:
    num_processes: int
    num_machines: int
    machine_rank: int
    main_process_ip: str
    main_process_port: int
    def __init__(self, num_processes, num_machines, machine_rank, main_process_ip, main_process_port) -> None: ...

def reconstruct_command_line(argv): ...

class PyTorchDistributedTask(Task):
    launch_cmd: Incomplete
    set_up_kwargs: Incomplete
    dist_args: Incomplete
    dist_env: Incomplete
    def __init__(self, launch_cmd: str, argv: list[str], slurm_args: SlurmArgs, verbose: bool = False, **set_up_kwargs) -> None: ...
    def dist_set_up(self): ...
    def command(self) -> str: ...
    def __call__(self) -> None: ...
