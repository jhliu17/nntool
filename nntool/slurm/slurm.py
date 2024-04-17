import sys
from typing import Any, Callable, Type, Union

import submitit

from .args import SlurmArgs
from ..parser import parse_from_cli


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
        gpus_per_node=slurm_config.gpus_per_node,
        timeout_min=slurm_config.timeout_min,
        slurm_additional_parameters=slurm_additional_parameters,
        **slurm_submission_kwargs,
    )

    return executor


def slurm_launcher(
    ArgsType: Type[Any],
    parser: Union[str, Callable] = "tyro",
    slurm_key: str = "slurm",
    slurm_distributed_env: bool = False,
    slurm_params_kwargs: dict = {},
    slurm_submit_kwargs: dict = {},
    *extra_args,
    **extra_kwargs,
):
    """A slurm launcher decorator

    :param ArgsType: the experiment arguments type, which should be a dataclass (it
                     mush have a slurm field)
    :return: decorator function with main entry
    """
    argv: list[str] = list(sys.argv)
    args = parse_from_cli(ArgsType, parser, *extra_args, **extra_kwargs)

    # check if args have slurm field
    if not hasattr(args, slurm_key):
        raise ValueError(
            f"ArgsType should have a field named `{slurm_key}` to use `slurm_launcher` decorator."
        )

    def decorator(main_fn):
        def wrapper():
            slurm_config = getattr(args, slurm_key)
            executor = get_slurm_executor(
                slurm_config,
                slurm_parameters_kwargs=slurm_params_kwargs,
                slurm_submission_kwargs=slurm_submit_kwargs,
            )
            job = executor.submit(main_fn, args)

            # get result to run program in debug mode
            if args.slurm.mode != "slurm":
                job.result()

        return wrapper

    return decorator
