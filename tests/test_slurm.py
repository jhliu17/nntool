import torch
import time
import accelerate
from nntool.slurm import (
    SlurmConfig,
    slurm_launcher,
    slurm_function,
)

distributed_slurm_settings = SlurmConfig(
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
    distributed_launch_command="accelerate launch --config_file tests/distributed.yaml --num_processes {num_processes} --num_machines {num_machines} --machine_rank {machine_rank} --main_process_ip {main_process_ip} --main_process_port {main_process_port} -m tests.test_slurm",
)

slurm_settings = SlurmConfig(
    mode="slurm",
    slurm_job_name="test_slurm",
    slurm_partition="zhanglab.p",
    node_list="laniakea",
    slurm_output_folder="tests/outputs/slurm",
    num_of_node=1,
    tasks_per_node=1,
    gpus_per_task=0,
    cpus_per_task=1,
    mem="2GB",
    timeout_min=10,
    use_distributed_env=False,
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


@slurm_function
def work_fn(a, b):
    """a demo function to test slurm"""
    return a + b


def test_distributed_slurm_function():
    job = distributed_fn(distributed_slurm_settings)(1, k=1)
    result = job.results()
    print(result)
    assert result == [0, 0]


def test_job_array_slurm_function():
    fn = work_fn(slurm_settings)

    job = fn(1, 2)
    result = job.result()
    print(result)
    assert result == 3

    jobs = fn.map_array([1, 2, 8, 9], [3, 4, 8, 9])
    results = [job.result() for job in jobs]
    print(results)
    assert results == [4, 6, 16, 18]


def test_slurm_launcher():
    # @slurm_launcher(SlurmConfig)
    # def slurm_main(args: SlurmConfig):
    #     """launch slurm to execute main function

    #     :param args: argument settings
    #     """
    #     print(args)
    #     run_job(30)
    #     return args

    pass
