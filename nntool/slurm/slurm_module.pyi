from ..parser import parse_from_cli as parse_from_cli
from .args import SlurmArgs as SlurmArgs
from .task import PyTorchDistributedTask as PyTorchDistributedTask
from dataclasses import dataclass
from submitit import Job as Job
from typing import Any, Callable

@dataclass
class SlurmFunction:
    """A slurm function for the slurm job, which can be used for distributed or non-distributed job (controlled by `use_distributed_env` in the slurm dataclass).

    **Exported Distributed Enviroment Variables**
    1. NNTOOL_SLURM_HAS_BEEN_SET_UP is a special environment variable to indicate that the slurm has been set up.
    2. After the set up, the distributed job will be launched and the following variables are exported:         num_processes: int, num_machines: int, machine_rank: int, main_process_ip: str, main_process_port: int.

    :param slurm_config: SlurmArgs, the slurm configuration dataclass, defaults to None
    :param slurm_params_kwargs: extra slurm arguments for the slurm configuration, defaults to {}
    :param slurm_submit_kwargs: extra slurm arguments for `srun` or `sbatch`, defaults to {}
    :param slurm_task_kwargs: extra arguments for the setting of distributed task, defaults to {}
    :param system_argv: the system arguments for the second launch in the distributed task (by default it will use the current system arguments `sys.argv[1:]`), defaults to None
    :param submit_fn: function to be submitted to Slurm, defaults to None
    :param default_submit_fn_args: default args for submit_fn, defaults to ()
    :param default_submit_fn_kwargs: default known word args for submit_fn, defaults to {}
    :return: the wrapped submit function with configured slurm paramters
    """
    slurm_config: SlurmArgs | None = ...
    slurm_params_kwargs: dict[str, Any] = ...
    slurm_submit_kwargs: dict[str, Any] = ...
    slurm_task_kwargs: dict[str, Any] = ...
    system_argv: list[str] | None = ...
    submit_fn: Callable[..., Any] | None = ...
    default_submit_fn_args: tuple[Any] = ...
    default_submit_fn_kwargs: dict[str, Any] = ...
    __doc__ = ...
    def __post_init__(self) -> None: ...
    def is_integrated(self):
        """Whether the slurm function has been set up.

        :return: True if the slurm function has been set up, False otherwise
        """
    def is_distributed(self):
        """Whether the slurm function is distributed.

        :return: True if the slurm function is distributed, False otherwise
        """
    @staticmethod
    def get_slurm_executor(slurm_config: SlurmArgs, slurm_parameters_kwargs: dict = {}, slurm_submission_kwargs: dict = {}): ...
    @staticmethod
    def slurm_has_been_set_up() -> bool:
        """NNTOOL_SLURM_HAS_BEEN_SET_UP is a special environment variable to indicate that the slurm has been set up.

        :return: bool
        """
    def update(self, slurm_config: SlurmArgs, slurm_params_kwargs: dict[str, Any] = {}, slurm_submit_kwargs: dict[str, Any] = {}, slurm_task_kwargs: dict[str, Any] = {}, system_argv: list[str] | None = None) -> SlurmFunction:
        """Update the slurm configuration for the slurm function.

        **Exported Distributed Enviroment Variables**
        1. NNTOOL_SLURM_HAS_BEEN_SET_UP is a special environment variable to indicate that the slurm has been set up.
        2. After the set up, the distributed job will be launched and the following variables are exported:         num_processes: int, num_machines: int, machine_rank: int, main_process_ip: str, main_process_port: int.

        :param slurm_config: SlurmArgs, the slurm configuration dataclass
        :param slurm_params_kwargs: extra slurm arguments for the slurm configuration, defaults to {}
        :param slurm_submit_kwargs: extra slurm arguments for `srun` or `sbatch`, defaults to {}
        :param slurm_task_kwargs: extra arguments for the setting of distributed task, defaults to {}
        :param system_argv: the system arguments for the second launch in the distributed task (by default it will use the current system arguments `sys.argv[1:]`), defaults to None
        :return: the wrapped submit function with configured slurm paramters
        """
    def __call__(self, *submit_fn_args, **submit_fn_kwargs) -> Job | Any:
        """Run the submit_fn with the given arguments and keyword arguments. The function is non-blocking in the mode of `slurm`, while other modes cause blocking. If there is no given arguments or keyword arguments, the default arguments and keyword arguments will be used.

        :raises ValueError: if the submit_fn is not set up
        :return: Slurm Job or the return value of the submit_fn
        """
    def __init__(self, slurm_config=..., slurm_params_kwargs=..., slurm_submit_kwargs=..., slurm_task_kwargs=..., system_argv=..., submit_fn=..., default_submit_fn_args=..., default_submit_fn_kwargs=...) -> None: ...

def slurm_launcher(ArgsType: type[Any], parser: str | Callable = 'tyro', slurm_key: str = 'slurm', slurm_params_kwargs: dict = {}, slurm_submit_kwargs: dict = {}, slurm_task_kwargs: dict = {}, *extra_args, **extra_kwargs) -> Callable[[Callable[..., Any]], SlurmFunction]:
    '''A slurm launcher decorator for distributed or non-distributed job (controlled by `use_distributed_env` in slurm field). This decorator should be used as the program entry. The decorated function is non-blocking in the mode of `slurm`, while other modes cause blocking.

    **Exported Distributed Enviroment Variables**
    1. NNTOOL_SLURM_HAS_BEEN_SET_UP is a special environment variable to indicate that the slurm has been set up.
    2. After the set up, the distributed job will be launched and the following variables are exported:         num_processes: int, num_machines: int, machine_rank: int, main_process_ip: str, main_process_port: int.

    :param ArgsType: the experiment arguments type, which should be a dataclass (it
                     mush have a slurm field defined by `slurm_key`)
    :param slurm_key: the key of the slurm field in the ArgsType, defaults to "slurm"
    :param parser: the parser for the arguments, defaults to "tyro"
    :param slurm_config: SlurmArgs, the slurm configuration dataclass
    :param slurm_params_kwargs: extra slurm arguments for the slurm configuration, defaults to {}
    :param slurm_submit_kwargs: extra slurm arguments for `srun` or `sbatch`, defaults to {}
    :param slurm_task_kwargs: extra arguments for the setting of distributed task, defaults to {}
    :param extra_args: extra arguments for the parser
    :param extra_kwargs: extra keyword arguments for the parser
    :return: decorator function with main entry
    '''
def slurm_distributed_launcher(ArgsType: type[Any], parser: str | Callable = 'tyro', slurm_key: str = 'slurm', slurm_params_kwargs: dict = {}, slurm_submit_kwargs: dict = {}, slurm_task_kwargs: dict = {}, *extra_args, **extra_kwargs) -> Callable[[Callable[..., Any]], SlurmFunction]:
    '''A slurm launcher decorator for the distributed job. This decorator should be used for the distributed job only and as the program entry. The decorated function is non-blocking in the mode of `slurm`, while other modes cause blocking.

    **Exported Distributed Enviroment Variables**
    1. NNTOOL_SLURM_HAS_BEEN_SET_UP is a special environment variable to indicate that the slurm has been set up.
    2. After the set up, the distributed job will be launched and the following variables are exported:         num_processes: int, num_machines: int, machine_rank: int, main_process_ip: str, main_process_port: int.

    :param ArgsType: the experiment arguments type, which should be a dataclass (it
                     mush have a slurm field defined by `slurm_key`)
    :param slurm_key: the key of the slurm field in the ArgsType, defaults to "slurm"
    :param parser: the parser for the arguments, defaults to "tyro"
    :param slurm_config: SlurmArgs, the slurm configuration dataclass
    :param slurm_params_kwargs: extra slurm arguments for the slurm configuration, defaults to {}
    :param slurm_submit_kwargs: extra slurm arguments for `srun` or `sbatch`, defaults to {}
    :param slurm_task_kwargs: extra arguments for the setting of distributed task, defaults to {}
    :param extra_args: extra arguments for the parser
    :param extra_kwargs: extra keyword arguments for the parser
    :return: decorator function with main entry
    '''
def slurm_function(submit_fn: Callable) -> Callable[..., SlurmFunction]:
    """A decorator to annoate a function to be run in slurm. The function decorated by this decorator should be launched in the way below.
    ```
    @slurm_function
    def run_in_slurm(*args, **kwargs):
        pass

    job = run_in_slurm(slurm_config)(*args, **kwargs)
    ```
    The decorated function `submit_fn` is non-blocking now. To block and get the return value, you can call `job.result()`.
    """
