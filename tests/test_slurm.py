import torch
import time
import accelerate
from nntool.slurm import (
    SlurmArgs,
    slurm_launcher,
    slurm_function,
)

slurm_settings = SlurmArgs(
    mode="slurm",
    slurm_job_name="test_slurm",
    slurm_partition="zhanglab.p",
    node_list="laniakea",
    slurm_output_folder="tests/outputs/slurm",
    num_of_node=1,
    tasks_per_node=2,
    gpus_per_task=1,
    cpus_per_task=1,
    mem="2GB",
    timeout_min=10,
    use_distributed_env=True,
    distributed_launch_command="accelerate launch --config_file distributed.yaml --num_processes {num_processes} --num_machines {num_machines} --machine_rank {machine_rank} --main_process_ip {main_process_ip} --main_process_port {main_process_port} -m tests.test_slurm",
)


def run_job(sleep_time: int = 30):
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    if accelerator.is_main_process:
        print("I am the main process")
        print(torch.cuda.device_count())
        a = torch.randn(1000, 1000).to(device)
        time.sleep(sleep_time)
    else:
        print("I am a worker process")
        print(torch.cuda.device_count())
        a = torch.randn(1000, 1000).to(device)
        time.sleep(sleep_time)


@slurm_function
def distributed_fn(*args, **kwargs):
    """a demo function to test slurm

    :param args: argument settings
    """
    print(args, kwargs)
    run_job(30)
    return args, kwargs


@slurm_launcher(SlurmArgs)
def slurm_main(args: SlurmArgs):
    """launch slurm to execute main function

    :param args: argument settings
    """
    print(args)
    run_job(30)
    return args


def test_slurm_function():
    job = distributed_fn(slurm_settings)(1, k=1)
    result = job.result()
    assert result == (1, {"k": 1})


def test_slurm_launcher():
    pass
