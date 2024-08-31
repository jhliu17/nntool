import sys
import os
import torch
import time
import accelerate
import pytest

from dataclasses import dataclass
from nntool.slurm import SlurmConfig, slurm_function, slurm_fn
from nntool.utils import get_current_time


def get_slurm_config(output_path, is_distributed: bool = False):
    slurm_config = None
    if is_distributed:
        slurm_config = SlurmConfig(
            mode="slurm",
            slurm_job_name="test_slurm",
            slurm_partition="zhanglab.p",
            node_list="galaxy",
            slurm_output_folder=f"{output_path}/{get_current_time()}/slurm",
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
    else:
        slurm_config = SlurmConfig(
            mode="slurm",
            slurm_job_name="test_slurm",
            slurm_partition="zhanglab.p",
            node_list="galaxy",
            slurm_output_folder=f"{output_path}/{get_current_time()}/slurm",
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
    return slurm_config


@dataclass
class WorkerTest:
    name: str

    @slurm_fn
    def run(self, a: int, b: int):
        print("My name is:", self.name)
        time.sleep(a + b)
        return a + b


def run_job(sleep_time: int = 30):
    print(torch.__file__)
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


@slurm_fn
def distributed_fn(*args, **kwargs):
    """a demo function to test slurm

    :param args: argument settings
    """
    print(args, kwargs)
    run_job(30)
    return args, kwargs


@slurm_fn
def work_fn(a, b):
    """a demo function to test slurm"""
    print(torch.__file__)
    print("PYTHONPATH", os.environ.get("PYTHONPATH"))
    time.sleep(a + b)
    return a + b


@pytest.mark.skipif(sys.platform != "linux", reason="Test only runs on Linux")
def test_distributed_slurm_function(tmp_path):
    slurm_settings = get_slurm_config("outputs/", is_distributed=True)
    job = distributed_fn[slurm_settings](1, k=1)
    result = job.results()
    print(result)
    assert result == [
        0,
    ]


@pytest.mark.skipif(sys.platform != "linux", reason="Test only runs on Linux")
def test_job_array_slurm_function(tmp_path):
    slurm_settings = get_slurm_config("outputs/", is_distributed=False)
    fn = work_fn[slurm_settings]

    job = fn(1, 2)
    result = job.result()
    print(result)
    assert result == 3

    jobs = fn.map_array([1, 2, 8, 9], [3, 4, 8, 9])
    results = [job.result() for job in jobs]
    print(results)
    assert results == [4, 6, 16, 18]


@pytest.mark.skipif(sys.platform != "linux", reason="Test only runs on Linux")
def test_sequential_jobs(tmp_path):
    slurm_settings = get_slurm_config("outputs/", is_distributed=False)

    jobs = []
    job1 = work_fn[slurm_settings](10, 2)
    jobs.append(job1)

    fn1 = work_fn[slurm_settings]
    fn1.on_condition(job1)
    job2 = fn1(7, 12)
    jobs.append(job2)

    fn2 = work_fn[slurm_settings]
    print(fn2.slurm_params_kwargs)
    assert fn1 is not fn2

    fn2.afterany(job1, job2)
    job3 = fn2(2, 30)
    jobs.append(job3)

    results = [job.result() for job in jobs]
    assert results == [12, 19, 32]


@pytest.mark.skipif(sys.platform != "linux", reason="Test only runs on Linux")
def test_class_slurm_function(tmp_path):
    worker = WorkerTest("test_worker")
    slurm_settings = get_slurm_config("outputs/", is_distributed=False)
    job = worker.run[slurm_settings](worker, 20, 10)
    result = job.result()
    print(result)
    assert result == 30


if __name__ == "__main__":
    run_job()
