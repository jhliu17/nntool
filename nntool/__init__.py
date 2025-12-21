from .version import VERSION, VERSION_SHORT
from .experiment import BaseExperimentConfig
from .slurm import slurm_fn, SlurmConfig
from .wandb import WandbConfig, init_wandb

__all__ = [
    "VERSION",
    "VERSION_SHORT",
    "BaseExperimentConfig",
    "slurm_fn",
    "SlurmConfig",
    "WandbConfig",
    "init_wandb",
]
__version__ = VERSION
