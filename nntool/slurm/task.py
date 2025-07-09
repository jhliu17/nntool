import os
import shlex
import shutil
import submitit

from pathlib import Path
from typing import Union, Generator, Callable
from dataclasses import dataclass
from .config import SlurmConfig
from .accelerator.utils import nvidia_smi_gpu_memory_stats_str

WANDB_DIRS = ("wandb", ".wandb")


def _is_py_or_dockerfile(path: str, root: str) -> bool:
    file = os.path.basename(path)
    return file.endswith(".py") or file.startswith("Dockerfile")


def include_code_files(path: str, root: str, code_ext: list[str]):
    file = os.path.basename(path)
    return any(file.endswith(ext) for ext in code_ext) or file.startswith("Dockerfile")


def exclude_code_folders(path: str, root: str, code_folders: list[str]):
    return any(
        os.path.relpath(path, root).startswith(code_folders + os.sep)
        for code_folders in code_folders
    )


def exclude_wandb_fn(path: str, root: str) -> bool:
    return any(
        os.path.relpath(path, root).startswith(wandb_dir + os.sep) for wandb_dir in WANDB_DIRS
    )


def filtered_dir(
    root: str,
    include_fn: Callable[[str, str], bool],
    exclude_fn: Callable[[str, str], bool],
) -> Generator[str, None, None]:
    """Simple generator to walk a directory."""

    for dirpath, _, files in os.walk(root):
        for fname in files:
            file_path = os.path.join(dirpath, fname)
            if include_fn(file_path, root) and not exclude_fn(file_path, root):
                yield file_path


def pack_code_files(
    root: str,
    target_root: str,
    include_fn: Callable[[str, str], bool] = _is_py_or_dockerfile,
    exclude_fn: Callable[[str, str], bool] = exclude_wandb_fn,
):
    root = os.path.abspath(root)
    code_root = Path(os.path.abspath(root))
    code_target = Path(os.path.abspath(target_root)) / "code"
    if not code_root.exists():
        raise ValueError(f"Code root {code_root} does not exist.")
    if not code_target.exists():
        code_target.mkdir(parents=True)

    for file_path in filtered_dir(root, include_fn, exclude_fn):
        save_name = os.path.relpath(file_path, root)
        sub_file_path, file_name = os.path.split(save_name)
        sub_file_full_path = code_target / sub_file_path
        if not sub_file_full_path.exists():
            sub_file_full_path.mkdir(parents=True)
        shutil.copy(file_path, sub_file_full_path / file_name)

    return code_target


def reconstruct_command_line(argv):
    # Quote each argument that needs special handling (like spaces or shell characters)
    # and join them with spaces to form the command line
    return " ".join(shlex.quote(arg) for arg in argv)


class Task:
    def __init__(self, argv: list[str], slurm_config: SlurmConfig, verbose: bool = False):
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
    num_processes: Union[int, str] = "$nntool_num_processes"
    num_machines: Union[int, str] = "$nntool_num_machines"
    machine_rank: Union[int, str] = "$nntool_machine_rank"
    main_process_ip: str = "$nntool_main_process_ip"
    main_process_port: Union[int, str] = "$nntool_main_process_port"

    def export_bash(self, output_folder: str):
        lines = ["#!/bin/bash"]
        for k, v in self.__dict__.items():
            lines.append(f"export nntool_{k}={v}")
        with open(os.path.join(output_folder, "nntool_distributed_env.sh"), "w") as f:
            f.write("\n".join(lines))


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
        self.dist_args = DistributedTaskConfig()
        self.dist_env = None

    def dist_set_up(self):
        self.log("running task on slurm")
        self.log("exporting PyTorch distributed environment variables")

        # prepare enviroment variables
        dist_env = submitit.helpers.TorchDistributedEnvironment().export(
            set_cuda_visible_devices=False
        )

        # other setup
        env_setup = {
            # "NCCL_DEBUG": "info",
            # "CUDA_LAUNCH_BLOCKING": "1",
        }

        # set CUDA visible devices if slurm has scheduled GPUs otherwise use all GPUs (without setting
        # CUDA_VISIBLE_DEVICES)
        env_setup.update(
            {"CUDA_VISIBLE_DEVICES": os.environ["SLURM_JOB_GPUS"]}
            if "SLURM_JOB_GPUS" in os.environ
            else {}
        )

        # other environment variables set by the user
        env_setup.update(self.env_setup_kwargs)
        self.log(f"env_setup: {env_setup}")

        # update environment variables
        os.environ.update(**env_setup)

        self.log(nvidia_smi_gpu_memory_stats_str())
        self.log(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        self.log(f"rank: {dist_env.rank}")
        self.log(f"world size: {dist_env.world_size}")
        self.log(f"local rank: {dist_env.local_rank}")
        self.log(f"local world size: {dist_env.local_world_size}")
        self.log(
            f"local rank {dist_env.local_rank}: CUDA_VISIBLE_DEVICES {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}"
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
        job_env = submitit.helpers.JobEnvironment()

        # concrete run command
        cmd = self.command()

        # export distributed environment variables
        if self.dist_env.local_rank == 0:
            print(f"running command: {cmd}")
            if self.slurm_config.mode == "slurm":
                try:
                    self.dist_args.export_bash(shlex.quote(str(job_env.paths.folder)))
                except Exception as e:
                    print(f"failed to export distributed environment variables: {e}")
                    return -1
            else:
                return os.system(cmd)

        return 0
