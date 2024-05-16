import cythonpackage

cythonpackage.init(__name__)

from .wandb_module import WandbConfig, init_wandb
