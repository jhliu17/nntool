import cythonpackage

cythonpackage.init(__name__)

from .args import SlurmArgs
from .slurm_module import slurm_function, slurm_launcher, slurm_distributed_launcher
from .task import PyTorchDistributedTask
