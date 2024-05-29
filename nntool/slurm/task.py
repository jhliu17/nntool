import os
import shlex

import submitit
from dataclasses import dataclass
from .args import SlurmConfig
from ..accelerator.utils import nvidia_smi_gpu_memory_stats_str


class Task:
    def __init__(
        self, argv: list[str], slurm_config: SlurmConfig, verbose: bool = False
    ):
        self.argv = argv
        self.slurm_config = slurm_config
        self.verbose = verbose

    def log(self, msg: str):
        if not self.verbose:
            return

        print(msg)

    def command(self) -> str:
        raise NotImplementedError

    def checkpoint(self):
        print("checkpointing")
        return submitit.helpers.DelayedSubmission(self)


@dataclass
class DistributedTaskConfig:
    num_processes: int
    num_machines: int
    machine_rank: int
    main_process_ip: str
    main_process_port: int

    def export_bash(self, output_folder: str):
        lines = ["#!/bin/bash"]
        for k, v in self.__dict__.items():
            lines.append(f"export nntool_{k}={v}")
        with open(os.path.join(output_folder, "distributed_env.sh"), "w") as f:
            f.write("\n".join(lines))


def reconstruct_command_line(argv):
    # Quote each argument that needs special handling (like spaces or shell characters)
    # and join them with spaces to form the command line
    return " ".join(shlex.quote(arg) for arg in argv)


class PyTorchDistributedTask(Task):
    """Ref:
    https://github.com/huggingface/accelerate/issues/1239
    https://github.com/yuvalkirstain/PickScore/blob/main/trainer/slurm_scripts/slurm_train.py
    https://github.com/facebookincubator/submitit/pull/1703
    """

    def __init__(
        self,
        launch_cmd: str,
        argv: list[str],
        slurm_config: SlurmConfig,
        verbose: bool = False,
        **env_setup_kwargs,
    ):
        super().__init__(argv, slurm_config, verbose)
        self.launch_cmd = launch_cmd
        self.env_setup_kwargs = env_setup_kwargs

        # to be set up in the dist_set_up method
        self.dist_args = DistributedTaskConfig(
            "$nntool_num_processes",
            "$nntool_num_machines",
            "$nntool_machine_rank",
            "$nntool_main_process_ip",
            "$nntool_main_process_port",
        )
        self.dist_env = None

    def dist_set_up(self):
        self.log("running task on slurm")
        self.log("exporting PyTorch distributed environment variables")

        # prepare enviroment variables
        dist_env = submitit.helpers.TorchDistributedEnvironment().export()

        # other setup
        env_setup = {
            # "CUDA_LAUNCH_BLOCKING": "1",
            # "NCCL_DEBUG": "info",
            "CUDA_VISIBLE_DEVICES": os.environ["SLURM_JOB_GPUS"],
        }
        env_setup.update(self.env_setup_kwargs)
        os.environ.update(**env_setup)

        self.log(nvidia_smi_gpu_memory_stats_str())
        self.log(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        self.log(f"rank: {dist_env.rank}")
        self.log(f"world size: {dist_env.world_size}")
        self.log(f"local rank: {dist_env.local_rank}")
        self.log(f"local world size: {dist_env.local_world_size}")
        self.log(
            f"local rank {dist_env.local_rank}: {os.environ['CUDA_VISIBLE_DEVICES']=}"
        )

        # set distributed arguments
        num_processes = (
            self.slurm_config.tasks_per_node
            * self.slurm_config.processes_per_task
            * self.slurm_config.num_of_node
        )
        machine_rank = dist_env.rank // self.slurm_config.tasks_per_node
        self.dist_args = DistributedTaskConfig(
            num_processes=num_processes,
            num_machines=self.slurm_config.num_of_node,
            machine_rank=machine_rank,
            main_process_ip=dist_env.master_addr,
            main_process_port=dist_env.master_port,
        )
        self.dist_env = dist_env

        return self.dist_args, self.dist_env

    def command(self) -> str:
        cmd = self.launch_cmd.format(**self.dist_args.__dict__)
        cmd += " " + reconstruct_command_line(self.argv)
        return cmd

    def __call__(self):
        # set up distributed environment
        self.dist_set_up()

        # job environment
        job = submitit.JobEnvironment()

        # run command
        cmd = self.command()

        if self.dist_env.local_rank == 0:
            print(f"running command: {cmd}")
            self.dist_args.export_bash(job.folder)
