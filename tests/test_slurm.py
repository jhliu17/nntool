import os
import torch
import time
import accelerate

from nntool.slurm import (
    SlurmConfig,
    slurm_function,
)
from nntool.utils import get_current_time
from nntool import test_import
from tests.src.my_file import test_import as my_test_import

distributed_slurm_settings = SlurmConfig(
    mode="slurm",
    slurm_job_name="test_slurm",
    slurm_partition="zhanglab.p",
    node_list="galaxy",
    slurm_output_folder=f"outputs/{get_current_time()}/slurm",
    num_of_node=1,
    tasks_per_node=1,
    gpus_per_task=2,
    cpus_per_task=1,
    mem="10G",
    timeout_min=10,
    pack_code=True,
    code_root="./",
    use_packed_code=True,
    use_distributed_env=True,
    processes_per_task=2,
    distributed_launch_command="accelerate launch --config_file tests/distributed.yaml --num_processes {num_processes} --num_machines {num_machines} --machine_rank {machine_rank} --main_process_ip {main_process_ip} --main_process_port {main_process_port} -m tests.test_slurm",
)

slurm_settings = SlurmConfig(
    mode="slurm",
    slurm_job_name="test_slurm",
    slurm_partition="zhanglab.p",
    node_list="galaxy",
    slurm_output_folder=f"outputs/{get_current_time()}/slurm",
    num_of_node=1,
    tasks_per_node=1,
    gpus_per_task=0,
    cpus_per_task=1,
    mem="2GB",
    timeout_min=10,
    pack_code=True,
    code_root="./",
    use_packed_code=True,
    use_distributed_env=False,
)


def run_job(sleep_time: int = 30):
    print(torch.__file__)
    test_import()
    my_test_import()
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
    print(torch.__file__)
    test_import()
    my_test_import()
    print("PYTHONPATH", os.environ.get("PYTHONPATH"))
    time.sleep(a + b)
    return a + b


def test_distributed_slurm_function():
    job = distributed_fn(distributed_slurm_settings)(1, k=1)
    result = job.results()
    print(result)
    assert result == [
        0,
    ]


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


def test_sequential_jobs():
    jobs = []
    job1 = work_fn(slurm_settings)(10, 2)
    jobs.append(job1)

    fn1 = work_fn(slurm_settings)
    fn1.on_condition(job1)
    job2 = fn1(7, 12)
    jobs.append(job2)

    fn2 = work_fn(slurm_settings)
    print(fn2.slurm_params_kwargs)
    assert fn1 is not fn2

    fn2.afterany(job1, job2)
    job3 = fn2(20, 30)
    jobs.append(job3)

    results = [job.result() for job in jobs]
    assert results == [12, 19, 50]


if __name__ == "__main__":
    run_job()
