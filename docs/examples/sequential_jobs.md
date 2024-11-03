# Using NNTool to chain sequential jobs

```py
import time
from nntool.slurm import SlurmConfig, slurm_fn

slurm_settings = SlurmConfig(
    mode="slurm",
    slurm_job_name="JOB_NAME",
    slurm_partition="PATITION",
    num_of_node=1,
    tasks_per_node=1,
    gpus_per_task=0,
    cpus_per_task=1,
    mem="2GB",
    timeout_min=10,
)


@slurm_fn
def work_fn(a, b):
    time.sleep(a + b)
    return a + b
```

```py
fn = work_fn[slurm_settings]

job = fn(1, 2)
result = job.result()
print(result)
assert result == 3

jobs = fn.map_array([1, 2, 8, 9], [3, 4, 8, 9])
results = [job.result() for job in jobs]
print(results)
assert results == [4, 6, 16, 18]
```

```py
jobs = []
job1 = work_fn[slurm_settings](10, 2)
jobs.append(job1)

fn1 = work_fn[slurm_settings]
fn1.on_condition(job1)
job2 = fn1(7, 12)
jobs.append(job2)

fn2 = work_fn[slurm_settings]
assert fn1 is not fn2

fn2.afterany(job1, job2)
job3 = fn2(2, 30)
jobs.append(job3)

results = [job.result() for job in jobs]
assert results == [12, 19, 32]
```
