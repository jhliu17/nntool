import os
import random

import submitit
from dataclasses import dataclass
from .args import SlurmArgs
from ..accelerator.utils import nvidia_smi_gpu_memory_stats


class Task:
    def __init__(self, argv: list[str], slurm_args: SlurmArgs, verbose: bool = False):
        self.argv = argv
        self.slurm_args = slurm_args
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
class DistributedArgs:
    num_processes: int
    num_machines: int
    machine_rank: int
    main_process_ip: str
    main_process_port: int


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
        slurm_args: SlurmArgs,
        verbose: bool = False,
    ):
        self.launch_cmd = launch_cmd
        self.argv = argv
        self.slurm_args = slurm_args
        self.verbose = verbose

        # to be set up in the dist_set_up method
        self.dist_args = DistributedArgs(None, None, None, None, None)
        self.dist_env = None

    def dist_set_up(self):
        self.log("running task on slurm")
        self.log("exporting PyTorch distributed environment variables")

        # prepare enviroment variables
        dist_env = submitit.helpers.TorchDistributedEnvironment()
        rng = random.Random(dist_env._job_env.job_id)
        dist_env.master_port = rng.randint(10000, 20000)
        dist_env = dist_env.export()
        os.environ.update(
            **{
                "CUDA_LAUNCH_BLOCKING": "1",
                "NCCL_DEBUG": "info",
                "CUDA_VISIBLE_DEVICES": os.environ["SLURM_JOB_GPUS"],
            }
        )

        self.log(nvidia_smi_gpu_memory_stats())
        self.log(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        self.log(f"rank: {dist_env.rank}")
        self.log(f"world size: {dist_env.world_size}")
        self.log(f"local rank: {dist_env.local_rank}")
        self.log(f"local world size: {dist_env.local_world_size}")
        self.log(
            f"local rank {dist_env.local_rank}: {os.environ['CUDA_VISIBLE_DEVICES']=}"
        )

        # set distributed arguments
        num_processes = self.slurm_args.tasks_per_node * self.slurm_args.num_of_node
        machine_rank = dist_env.rank // self.slurm_args.tasks_per_node
        self.dist_args = DistributedArgs(
            num_processes=num_processes,
            num_machines=self.slurm_args.num_of_node,
            machine_rank=machine_rank,
            main_process_ip=dist_env.master_addr,
            main_process_port=dist_env.master_port,
        )
        self.dist_env = dist_env

        return self.dist_args, self.dist_env

    def command(self) -> str:
        return self.launch_cmd

    def __call__(self):
        # set up distributed environment
        self.dist_set_up()

        # run command
        cmd = self.command()

        if self.dist_env.local_rank == 0:
            print(f"running command: {cmd}")
            exit_code = os.system(cmd)
        else:
            exit_code = 0
            print("waiting for master to finish")

        if exit_code != 0:
            raise RuntimeError(f"command {cmd} failed with exit code {exit_code}")
