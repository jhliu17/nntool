import sys
from dataclasses import dataclass
from typing import Any, Callable, Literal, Type, Union


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
    num_of_node: int = 1

    # tasks per node
    tasks_per_node: int = 1

    # number of gpus per node to request
    gpus_per_node: int = 0

    # number of cpus per task to request
    cpus_per_task: int = 1

    # memory to request (leave black to use default memory configurations in the node)
    mem: str = ""

    # time out min
    timeout_min: int = sys.maxsize
