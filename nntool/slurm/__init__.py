import cythonpackage

cythonpackage.init(__name__)

from .config import SlurmConfig, SlurmArgs
from .wrap import (
    slurm_function,
    slurm_fn,
    slurm_launcher,
    slurm_distributed_launcher,
)
from .task import PyTorchDistributedTask
