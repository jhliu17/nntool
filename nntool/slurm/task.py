import os
import random

import submitit

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

    def command(self):
        is_python_module = not self.argv[0].endswith(".py")

        cmd = ""
        if is_python_module:
            cmd = f"-m {' '.join(self.argv)}"
        else:
            cmd = " ".join(self.argv)
        return cmd

    def checkpoint(self):
        print("checkpointing")
        return submitit.helpers.DelayedSubmission(self)


class PyTorchDistributedTask(Task):
    """Ref:
    https://github.com/huggingface/accelerate/issues/1239
    https://github.com/yuvalkirstain/PickScore/blob/main/trainer/slurm_scripts/slurm_train.py
    https://github.com/facebookincubator/submitit/pull/1703

    """

    def __call__(self):
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

        num_processes = self.cfg.slurm.n_processes * self.cfg.slurm.n_nodes
        machine_rank = dist_env.rank // self.cfg.slurm.n_processes
        cmd = f"accelerate launch --dynamo_backend no --num_processes {num_processes} --num_machines {self.cfg.slurm.n_nodes} --use_deepspeed --machine_rank {machine_rank} --main_process_ip {dist_env.master_addr} --main_process_port {dist_env.master_port} trainer/scripts/train.py {self.cfg.slurm.cmd}"

        self.log(f"running command:\n{cmd}")

        if dist_env.local_rank == 0:
            os.system(cmd)
        else:
            self.log("Waiting for master to finish")
