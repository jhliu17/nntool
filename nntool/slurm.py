import sys
import submitit

from typing import Union, Callable, Type, Any
from dataclasses import dataclass
from typing import Literal
from .parser import parse_from_cli


@dataclass
class SlurmArgs:
    # running mode
    mode: Literal["debug", "local", "slurm"] = "debug"

    # slurm job name
    slurm_job_name: str = "YOUR_JOB_NAME"

    # slurm partition name
    slurm_partition: str = "YOUR_PARTITION_NAME"

    # slurm output folder
    slurm_output_folder: str = "outputs/slurm"

    # node list string (leave blank to use all nodes)
    node_list: str = ""

    # node list string to be excluded (leave blank to use all nodes in the node list)
    node_list_exclude: str = ""

    # number of nodes to request
    req_num_of_node: int = 1

    # tasks per node
    tasks_per_node: int = 1

    # number of gpus per node to request
    gpus_per_node: int = 0

    # number of cpus per task to request
    cpus_per_task: int = 1

    # memory to request (leave black to use all memory in the node)
    mem: str = ""

    # time out min
    timeout_min: int = sys.maxsize


def get_slurm_executor(slurm_config: SlurmArgs):
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

    # set slurm parameters
    executor.update_parameters(
        name=slurm_config.slurm_job_name,
        slurm_partition=slurm_config.slurm_partition,
        nodes=slurm_config.req_num_of_node,
        tasks_per_node=slurm_config.tasks_per_node,
        cpus_per_task=slurm_config.cpus_per_task,
        gpus_per_node=slurm_config.gpus_per_node,
        timeout_min=slurm_config.timeout_min,
        slurm_additional_parameters=slurm_additional_parameters,
    )

    return executor


def slurm_launcher(
    ArgsType: Type[Any],
    slurm_key: str = "slurm",
    parser: Union[str, Callable] = "tyro",
    *args,
    **kwargs,
):
    """A slurm launcher decorator

    :param ArgsType: the experiment arguments type, which should be a dataclass (it
                     mush have a slurm field)
    :return: decorator function with main entry
    """
    args = parse_from_cli(ArgsType, parser, *args, **kwargs)

    # check if args have slurm field
    if not hasattr(args, slurm_key):
        raise ValueError(
            f"ArgsType should have a field named `{slurm_key}` to use `slurm_launcher` decorator."
        )

    def decorator(main_fn):
        def wrapper():
            slurm_config = getattr(args, slurm_key)
            executor = get_slurm_executor(slurm_config)
            job = executor.submit(main_fn, args)

            # get result to run program in debug mode
            if args.slurm.mode != "slurm":
                job.result()

        return wrapper

    return decorator
