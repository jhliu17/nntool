import os
import sys
from functools import wraps
from typing import Any, Callable, Type, Union

import submitit

from ..parser import parse_from_cli
from .args import SlurmArgs
from .task import PyTorchDistributedTask


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
        tasks_per_node=slurm_config.tasks_per_node,
        cpus_per_task=slurm_config.cpus_per_task,
        gpus_per_task=slurm_config.gpus_per_task,
        timeout_min=slurm_config.timeout_min,
        slurm_additional_parameters=slurm_additional_parameters,
        **slurm_submission_kwargs,
    )

    return executor


def slurm_launcher(
    ArgsType: Type[Any],
    parser: Union[str, Callable] = "tyro",
    slurm_key: str = "slurm",
    slurm_params_kwargs: dict = {},
    slurm_submit_kwargs: dict = {},
    *extra_args,
    **extra_kwargs,
):
    """A slurm launcher decorator

    NNTOOL_SLURM_HAS_BEEN_SET_UP is a special environment variable to indicate that the slurm has been set up.

    :param ArgsType: the experiment arguments type, which should be a dataclass (it
                     mush have a slurm field)
    :return: decorator function with main entry
    """
    sys_argv = list(sys.argv[1:])
    sys_arg_str = " ".join(sys_argv)
    args = parse_from_cli(ArgsType, parser, *extra_args, **extra_kwargs)

    # check whether slurm has been set up
    slurm_has_been_set_up = False
    if os.environ.get("NNTOOL_SLURM_HAS_BEEN_SET_UP") is not None:
        slurm_has_been_set_up = True

    # check if args have slurm field
    if not hasattr(args, slurm_key):
        raise ValueError(
            f"ArgsType should have a field named `{slurm_key}` to use `slurm_launcher` decorator."
        )
    slurm_config: SlurmArgs = getattr(args, slurm_key)

    def decorator(main_fn):
        def wrapper():
            if not slurm_config.distributed_env:
                executor = get_slurm_executor(
                    slurm_config,
                    slurm_parameters_kwargs=slurm_params_kwargs,
                    slurm_submission_kwargs=slurm_submit_kwargs,
                )
                job = executor.submit(main_fn, args)

                # get result to run program in debug mode
                if slurm_config.mode != "slurm":
                    job.result()
            else:
                if not slurm_has_been_set_up:
                    executor = get_slurm_executor(
                        slurm_config,
                        slurm_parameters_kwargs=slurm_params_kwargs,
                        slurm_submission_kwargs=slurm_submit_kwargs,
                    )

                    # prepare distributed env for the second launch
                    task = PyTorchDistributedTask(
                        f"export NNTOOL_SLURM_HAS_BEEN_SET_UP=1; {slurm_config.distributed_launch_command} {sys_arg_str}",
                        sys_argv,
                        slurm_config,
                        verbose=True,
                    )

                    job = executor.submit(task)

                    # get result to run program in debug mode
                    if slurm_config.mode != "slurm":
                        job.result()
                else:
                    main_fn(args)

        return wrapper

    return decorator


def slurm_function(
    submit_fn: Callable,
):
    """A decorator to annoate a function to be run in slurm"""

    @wraps(submit_fn)
    def wrapper(
        slurm_config: SlurmArgs,
        slurm_params_kwargs: dict = {},
        slurm_submit_kwargs: dict = {},
        *to_run_args,
        **to_run_kwargs,
    ):
        executor = get_slurm_executor(
            slurm_config,
            slurm_parameters_kwargs=slurm_params_kwargs,
            slurm_submission_kwargs=slurm_submit_kwargs,
        )
        job = executor.submit(submit_fn, *to_run_args, **to_run_kwargs)

        # get result to run program in debug mode
        if slurm_config.mode != "slurm":
            job.result()

    return wrapper
