from typing import Literal

class SlurmArgs:
    mode: Literal['debug', 'local', 'slurm']
    slurm_job_name: str
    slurm_partition: str
    slurm_output_folder: str
    node_list: str
    node_list_exclude: str
    num_of_node: int
    tasks_per_node: int
    gpus_per_task: int
    cpus_per_task: int
    gpus_per_node: int | None
    mem: str
    timeout_min: int
    use_distributed_env: bool
    distributed_launch_command: str
