import cythonpackage

cythonpackage.init(__name__)

from .args import SlurmConfig, SlurmArgs
from .slurm_module import (
    slurm_function,
    slurm_fn,
    slurm_launcher,
    slurm_distributed_launcher,
)
from .task import PyTorchDistributedTask
