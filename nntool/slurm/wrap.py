import sys

from warnings import warn
from typing import Any, Callable, Type, Union, Dict, List

from .config import SlurmConfig
from .function import SlurmFunction
from ..parser import parse_from_cli


def slurm_fn(
    submit_fn: Callable,
) -> SlurmFunction:
    """A decorator to annoate a function to be run on slurm. The function decorated by this decorator should be launched on the way below. The decorated function `submit_fn` is non-blocking now. To block and get the return value, you can call `job.result()`.

    **Example**

    Here's an example of how to use this function:

    ``` py
    @slurm_fn
    def run_on_slurm(a, b):
        return a + b

    slurm_config = SlurmConfig(
        mode="slurm",
        slurm_partition="PARTITION",
        slurm_job_name="EXAMPLE",
        tasks_per_node=1,
        cpus_per_task=8,
        mem="1GB",
    )
    job = run_on_slurm[slurm_config](1, b=2)
    result = job.result()  # block and get the result
    ```

    :param submit_fn: the function to be run on slurm
    :return: the function to be run on slurm
    """
    slurm_fn = SlurmFunction(submit_fn=submit_fn)

    return slurm_fn


def slurm_launcher(
    ArgsType: Type[Any],
    parser: Union[str, Callable] = "tyro",
    slurm_key: str = "slurm",
    slurm_params_kwargs: dict = {},
    slurm_submit_kwargs: dict = {},
    slurm_task_kwargs: dict = {},
    *extra_args,
    **extra_kwargs,
) -> Callable[[Callable[..., Any]], SlurmFunction]:
    """A slurm launcher decorator for distributed or non-distributed job (controlled by `use_distributed_env` in slurm field). This decorator should be used as the program entry. The decorated function is non-blocking in the mode of `slurm`, while other modes cause blocking.

    #### Exported Distributed Enviroment Variables
    1. NNTOOL_SLURM_HAS_BEEN_SET_UP is a special environment variable to indicate that the slurm has been set up.
    2. After the set up, the distributed job will be launched and the following variables are exported:         num_processes: int, num_machines: int, machine_rank: int, main_process_ip: str, main_process_port: int.

    :param ArgsType: the experiment arguments type, which should be a dataclass (it
                     mush have a slurm field defined by `slurm_key`)
    :param slurm_key: the key of the slurm field in the ArgsType, defaults to "slurm"
    :param parser: the parser for the arguments, defaults to "tyro"
    :param slurm_config: SlurmConfig, the slurm configuration dataclass
    :param slurm_params_kwargs: extra slurm arguments for the slurm configuration, defaults to {}
    :param slurm_submit_kwargs: extra slurm arguments for `srun` or `sbatch`, defaults to {}
    :param slurm_task_kwargs: extra arguments for the setting of distributed task, defaults to {}
    :param extra_args: extra arguments for the parser
    :param extra_kwargs: extra keyword arguments for the parser
    :return: decorator function with main entry
    """
    argv = list(sys.argv[1:])
    args = parse_from_cli(ArgsType, parser, *extra_args, **extra_kwargs)

    # check if args have slurm field
    if not hasattr(args, slurm_key):
        raise ValueError(
            f"ArgsType should have a field named `{slurm_key}` to use `slurm_launcher` decorator."
        )
    slurm_config: SlurmConfig = getattr(args, slurm_key)

    def decorator(
        submit_fn: Callable[..., Any],
    ) -> SlurmFunction:
        return SlurmFunction(
            submit_fn=submit_fn,
            default_submit_fn_args=(args,),
        ).configure(
            slurm_config,
            slurm_params_kwargs,
            slurm_submit_kwargs,
            slurm_task_kwargs,
            system_argv=argv,
        )

    return decorator


def slurm_distributed_launcher(
    ArgsType: Type[Any],
    parser: Union[str, Callable] = "tyro",
    slurm_key: str = "slurm",
    slurm_params_kwargs: dict = {},
    slurm_submit_kwargs: dict = {},
    slurm_task_kwargs: dict = {},
    *extra_args,
    **extra_kwargs,
) -> Callable[[Callable[..., Any]], SlurmFunction]:
    """A slurm launcher decorator for the distributed job. This decorator should be used for the distributed job only and as the program entry. The decorated function is non-blocking in the mode of `slurm`, while other modes cause blocking.

    #### Exported Distributed Enviroment Variables
    1. NNTOOL_SLURM_HAS_BEEN_SET_UP is a special environment variable to indicate that the slurm has been set up.
    2. After the set up, the distributed job will be launched and the following variables are exported:         num_processes: int, num_machines: int, machine_rank: int, main_process_ip: str, main_process_port: int.

    :param ArgsType: the experiment arguments type, which should be a dataclass (it
                     mush have a slurm field defined by `slurm_key`)
    :param slurm_key: the key of the slurm field in the ArgsType, defaults to "slurm"
    :param parser: the parser for the arguments, defaults to "tyro"
    :param slurm_config: SlurmConfig, the slurm configuration dataclass
    :param slurm_params_kwargs: extra slurm arguments for the slurm configuration, defaults to {}
    :param slurm_submit_kwargs: extra slurm arguments for `srun` or `sbatch`, defaults to {}
    :param slurm_task_kwargs: extra arguments for the setting of distributed task, defaults to {}
    :param extra_args: extra arguments for the parser
    :param extra_kwargs: extra keyword arguments for the parser
    :return: decorator function with main entry
    """
    warn(
        "`slurm_distributed_launcher` has been deprecated. Please use `slurm_launcher` instead, which supports both distributed and non-distributed job (controlled by `use_distributed_env` in slurm field).",
        DeprecationWarning,
        stacklevel=2,
    )
    argv = list(sys.argv[1:])
    args = parse_from_cli(ArgsType, parser, *extra_args, **extra_kwargs)

    # check if args have slurm field
    if not hasattr(args, slurm_key):
        raise ValueError(
            f"ArgsType should have a field named `{slurm_key}` to use `slurm_distributed_launcher` decorator."
        )
    slurm_config: SlurmConfig = getattr(args, slurm_key)

    def decorator(
        submit_fn: Callable[..., Any],
    ) -> SlurmFunction:
        return SlurmFunction(
            submit_fn=submit_fn,
            default_submit_fn_args=(args,),
        ).configure(
            slurm_config,
            slurm_params_kwargs,
            slurm_submit_kwargs,
            slurm_task_kwargs,
            system_argv=argv,
        )

    return decorator


def slurm_function(
    submit_fn: Callable,
):
    """A decorator to annoate a function to be run in slurm. The function decorated by this decorator should be launched in the way below.
    ```
    @slurm_function
    def run_in_slurm(*args, **kwargs):
        pass

    job = run_in_slurm(slurm_config)(*args, **kwargs)
    ```
    The decorated function `submit_fn` is non-blocking now. To block and get the return value, you can call `job.result()`.
    """

    def wrapper(
        slurm_config: SlurmConfig,
        slurm_params_kwargs: Dict[str, Any] = {},
        slurm_submit_kwargs: Dict[str, Any] = {},
        slurm_task_kwargs: Dict[str, Any] = {},
        system_argv: Union[List[str], None] = None,
    ) -> SlurmFunction:
        """Update the slurm configuration for the slurm function.

        #### Exported Distributed Enviroment Variables
        1. NNTOOL_SLURM_HAS_BEEN_SET_UP is a special environment variable to indicate that the slurm has been set up.
        2. After the set up, the distributed job will be launched and the following variables are exported:         num_processes: int, num_machines: int, machine_rank: int, main_process_ip: str, main_process_port: int.

        :param slurm_config: SlurmConfig, the slurm configuration dataclass
        :param slurm_params_kwargs: extra slurm arguments for the slurm configuration, defaults to {}
        :param slurm_submit_kwargs: extra slurm arguments for `srun` or `sbatch`, defaults to {}
        :param slurm_task_kwargs: extra arguments for the setting of distributed task, defaults to {}
        :param system_argv: the system arguments for the second launch in the distributed task (by default it will use the current system arguments `sys.argv[1:]`), defaults to None
        :return: the wrapped submit function with configured slurm paramters
        """
        warn(
            "`slurm_function` has been deprecated. Please use `slurm_fn` instead, which supports both distributed and non-distributed job (controlled by `use_distributed_env` in slurm field).",
            DeprecationWarning,
            stacklevel=2,
        )
        slurm_fn = SlurmFunction(
            submit_fn=submit_fn,
        ).configure(
            slurm_config,
            slurm_params_kwargs,
            slurm_submit_kwargs,
            slurm_task_kwargs,
            system_argv,
        )
        return slurm_fn

    return wrapper
