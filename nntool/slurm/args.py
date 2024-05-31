import sys
from dataclasses import dataclass, field
from typing import Literal, Union, Dict


@dataclass
class SlurmConfig:
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

    # number of gpus per task to request
    gpus_per_task: int = 0

    # number of cpus per task to request
    cpus_per_task: int = 1

    # number of gpus per node to request (if this is set, gpus_per_task will be ignored)
    gpus_per_node: Union[int, None] = None

    # memory to request (leave black to use default memory configurations in the node)
    mem: str = ""

    # time out min
    timeout_min: int = sys.maxsize

    # whether to pack code
    pack_code: bool = True

    # use packed code to run
    use_packed_code: bool = False

    # code root
    code_root: str = "."

    # code file extensions
    code_file_suffixes: list[str] = field(
        default_factory=lambda: [".py", ".sh", ".yaml", ".toml"]
    )

    # exclude folders (relative to the code root)
    exclude_code_folders: list[str] = field(
        default_factory=lambda: ["wandb", "outputs", "datasets"]
    )

    # whether to use distributed environment
    use_distributed_env: bool = False

    # processes per task (this value is not used by slurm, but in the distributed environment)
    processes_per_task: int = 1

    # distributed launch command (this will be called after the distributed enviroment is set up)
    # the following environment variables are available:
    #   num_processes: int
    #   num_machines: int
    #   machine_rank: int
    #   main_process_ip: str
    #   main_process_port: int
    # use braces to access the environment variables, e.g. {num_processes}
    distributed_launch_command: str = ""

    # extra slurm job parameters
    slurm_params_kwargs: Dict[str, str] = field(default_factory=dict)

    # extra slurm submit parameters
    slurm_submit_kwargs: Dict[str, str] = field(default_factory=dict)

    # extra slurm task parameters
    slurm_task_kwargs: Dict[str, str] = field(default_factory=dict)


SlurmArgs = SlurmConfig
