import os
import sys
from warnings import warn
from dataclasses import dataclass
from typing import Any, Callable, Type, Union, Dict, List

import submitit

from ..parser import parse_from_cli
from .args import SlurmArgs
from .task import PyTorchDistributedTask


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
    :param default_submit_fn_args: default args for submit_fn, defaults to None
    :param default_submit_fn_kwargs: default known word args for submit_fn, defaults to None
    :return: the wrapped submit function with configured slurm paramters
    """

    slurm_config: SlurmArgs = None
    slurm_params_kwargs: Dict[str, Any] = {}
    slurm_submit_kwargs: Dict[str, Any] = {}
    slurm_task_kwargs: Dict[str, Any] = {}
    system_argv: Union[List[str], None] = None
    submit_fn: Union[Callable[..., Any], None] = None
    default_submit_fn_args: Union[List[Any], None] = None
    default_submit_fn_kwargs: Union[Dict[str, Any], None] = None

    @staticmethod
    def get_slurm_executor(
        slurm_config: SlurmArgs,
        slurm_parameters_kwargs: dict = {},
        slurm_submission_kwargs: dict = {},
    ):
        executor = submitit.AutoExecutor(
            folder=slurm_config.slurm_output_folder,
            cluster=None if slurm_config.mode == "slurm" else slurm_config.mode,
        )

        # set additional parameters
        slurm_additional_parameters = {}
        if slurm_config.node_list:
            slurm_additional_parameters["nodelist"] = slurm_config.node_list
        if slurm_config.node_list_exclude:
            slurm_additional_parameters["exclude"] = slurm_config.node_list_exclude
        if slurm_config.mem:
            slurm_additional_parameters["mem"] = slurm_config.mem
        slurm_additional_parameters.update(slurm_parameters_kwargs)

        # set slurm parameters
        executor.update_parameters(
            name=slurm_config.slurm_job_name,
            slurm_partition=slurm_config.slurm_partition,
            nodes=slurm_config.num_of_node,
            slurm_tasks_per_node=slurm_config.tasks_per_node,
            slurm_cpus_per_task=slurm_config.cpus_per_task,
            slurm_gpus_per_node=(
                slurm_config.gpus_per_task * slurm_config.tasks_per_node
                if slurm_config.gpus_per_node is None
                else slurm_config.gpus_per_node
            ),  # gpu cannot be assigned in the task level
            timeout_min=slurm_config.timeout_min,
            slurm_additional_parameters=slurm_additional_parameters,
            **slurm_submission_kwargs,
        )

        return executor

    @staticmethod
    def slurm_has_been_set_up() -> bool:
        """NNTOOL_SLURM_HAS_BEEN_SET_UP is a special environment variable to indicate that the slurm has been set up.

        :return: bool
        """
        # check whether slurm has been set up
        has_been_set_up = False
        if os.environ.get("NNTOOL_SLURM_HAS_BEEN_SET_UP") is not None:
            has_been_set_up = True
        return has_been_set_up

    def update(
        self,
        slurm_config: SlurmArgs,
        slurm_params_kwargs: Dict[str, Any] = {},
        slurm_submit_kwargs: Dict[str, Any] = {},
        slurm_task_kwargs: Dict[str, Any] = {},
        system_argv: Union[List[str], None] = None,
    ) -> "SlurmFunction":
        """A slurm function for the slurm job, which can be used for distributed or non-distributed job (controlled by `use_distributed_env` in the slurm dataclass).

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
        self.slurm_config = slurm_config
        self.slurm_params_kwargs = slurm_params_kwargs
        self.slurm_submit_kwargs = slurm_submit_kwargs
        self.slurm_task_kwargs = slurm_task_kwargs
        self.system_argv = system_argv
        return self

    def __call__(self, *submit_fn_args, **submit_fn_kwargs) -> Any:
        if self.slurm_config is None:
            raise ValueError("Slurm function should be initialized before calling.")

        if not self.slurm_config.use_distributed_env:
            return self._submit(*submit_fn_args, **submit_fn_kwargs)
        else:
            return self._distributed_submit(*submit_fn_args, **submit_fn_kwargs)

    def _submit(
        self,
        *submit_fn_args,
        **submit_fn_kwargs,
    ) -> Any:
        submit_fn_args = (
            self.default_submit_fn_args if not submit_fn_args else submit_fn_args
        )
        submit_fn_kwargs = (
            self.default_submit_fn_kwargs if not submit_fn_kwargs else submit_fn_kwargs
        )

        executor = self.get_slurm_executor(
            self.slurm_config,
            slurm_parameters_kwargs=self.slurm_params_kwargs,
            slurm_submission_kwargs=self.slurm_submit_kwargs,
        )
        job = executor.submit(self.submit_fn, *submit_fn_args, **submit_fn_kwargs)

        # get result to run program in debug mode
        if self.slurm_config.mode != "slurm":
            job.result()

        return job

    def _distributed_submit(
        self,
        *submit_fn_args,
        **submit_fn_kwargs,
    ) -> Any:
        submit_fn_args = (
            self.default_submit_fn_args if not submit_fn_args else submit_fn_args
        )
        submit_fn_kwargs = (
            self.default_submit_fn_kwargs if not submit_fn_kwargs else submit_fn_kwargs
        )

        if not self.slurm_has_been_set_up():
            executor = self.get_slurm_executor(
                self.slurm_config,
                slurm_parameters_kwargs=self.slurm_params_kwargs,
                slurm_submission_kwargs=self.slurm_submit_kwargs,
            )

            # prepare distributed env for the second launch
            task = PyTorchDistributedTask(
                f"export NNTOOL_SLURM_HAS_BEEN_SET_UP=1; {self.slurm_config.distributed_launch_command}",
                (
                    self.system_argv
                    if self.system_argv is not None
                    else list(sys.argv[1:])
                ),
                self.slurm_config,
                verbose=True,
                **self.slurm_task_kwargs,
            )

            job = executor.submit(task)

            # get result to run program in debug mode
            if self.slurm_config.mode != "slurm":
                job.result()

            return job
        else:
            return self.submit_fn(*submit_fn_args, **submit_fn_kwargs)


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
    """
    argv = list(sys.argv[1:])
    args = parse_from_cli(ArgsType, parser, *extra_args, **extra_kwargs)

    # check if args have slurm field
    if not hasattr(args, slurm_key):
        raise ValueError(
            f"ArgsType should have a field named `{slurm_key}` to use `slurm_launcher` decorator."
        )
    slurm_config: SlurmArgs = getattr(args, slurm_key)

    def decorator(
        submit_fn: Callable[..., Any],
    ) -> SlurmFunction:
        return SlurmFunction(
            slurm_config,
            slurm_params_kwargs,
            slurm_submit_kwargs,
            slurm_task_kwargs,
            system_argv=argv,
            submit_fn=submit_fn,
            default_submit_fn_args=args,
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
) -> SlurmFunction:
    """A slurm launcher decorator for the distributed job. This decorator should be used for the distributed job only and as the program entry. The decorated function is non-blocking in the mode of `slurm`, while other modes cause blocking.

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
    """
    warn(
        "`slurm_distributed_launcher` has been deprecated. Please use `slurm_launcher` instead, which supports both distributed or non-distributed job (controlled by `use_distributed_env` in slurm field).",
        DeprecationWarning,
        stacklevel=2,
    )
    argv = list(sys.argv[1:])
    args = parse_from_cli(ArgsType, parser, *extra_args, **extra_kwargs)

    # check if args have slurm field
    if not hasattr(args, slurm_key):
        raise ValueError(
            f"ArgsType should have a field named `{slurm_key}` to use `slurm_launcher` decorator."
        )
    slurm_config: SlurmArgs = getattr(args, slurm_key)

    def decorator(
        submit_fn: Callable[..., Any],
    ) -> SlurmFunction:
        return SlurmFunction(
            slurm_config,
            slurm_params_kwargs,
            slurm_submit_kwargs,
            slurm_task_kwargs,
            system_argv=argv,
            submit_fn=submit_fn,
            default_submit_fn_args=args,
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
    return SlurmFunction(submit_fn=submit_fn).update
