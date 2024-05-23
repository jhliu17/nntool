import torch
import time
import accelerate
import tyro
from nntool.slurm import (
    SlurmArgs,
    slurm_launcher,
    slurm_function,
)
from dataclasses import dataclass


@dataclass
class ExperimentArgs:
    slurm: SlurmArgs

    experiment_name: str = "test_slurm"


@slurm_function
def distributed_fn(args: ExperimentArgs):
    """a demo function to test slurm

    :param args: argument settings
    """
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    if accelerator.is_main_process:
        print("I am the main process")
        print(torch.cuda.device_count())
        print(args)
        a = torch.randn(1000, 1000).to(device)
        time.sleep(60)
    else:
        print("I am a worker process")
        print(torch.cuda.device_count())
        a = torch.randn(1000, 1000).to(device)
        time.sleep(60)


@slurm_launcher(ExperimentArgs)
def slurm_main(args: ExperimentArgs):
    """launch slurm to execute main function

    :param args: argument settings
    """
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    if accelerator.is_main_process:
        print("I am the main process")
        print(torch.cuda.device_count())
        print(args)
        a = torch.randn(1000, 1000).to(device)
        time.sleep(60)
    else:
        print("I am a worker process")
        print(torch.cuda.device_count())
        a = torch.randn(1000, 1000).to(device)
        time.sleep(60)


def main():
    args = tyro.cli(ExperimentArgs)
    fn = distributed_fn(args.slurm)
    fn(args)


if __name__ == "__main__":
    main()
    # slurm_main()
