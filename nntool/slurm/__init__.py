from slurmic import SlurmConfig, SlurmArgs
from slurmic.function import SlurmFunction
from slurmic.task import Task, DistributedTaskConfig, PyTorchDistributedTask
from slurmic.wrap import slurm_fn, slurm_function, slurm_launcher


__all__ = [
    "SlurmConfig",
    "SlurmArgs",
    "SlurmFunction",
    "slurm_fn",
    "slurm_function",
    "slurm_launcher",
    "Task",
    "DistributedTaskConfig",
    "PyTorchDistributedTask",
]
