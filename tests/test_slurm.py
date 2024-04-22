import torch
import time
import accelerate
from nntool.slurm import SlurmArgs, slurm_launcher
from dataclasses import dataclass


@dataclass
class ExperimentArgs:
    slurm: SlurmArgs

    experiment_name: str = "slurm_experiment_name"


@slurm_launcher(ExperimentArgs)
def main(args: ExperimentArgs):
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


if __name__ == "__main__":
    main()
