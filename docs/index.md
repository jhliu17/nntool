---
hide:
  - navigation
  - toc
---

# Welcome to 🚂 NNTool

Using NNTool to submit a Python function to SLURM

## Installation

```bash
pip install nntool
```

## Quick Example

``` py title="Create a Python function to be executed on Slurm" hl_lines="3 7-15"
  from nntool.slurm import SlurmConfig, slurm_fn

  @slurm_fn
  def run_on_slurm(a, b):
      return a + b

  slurm_config = SlurmConfig(
      mode="slurm",
      slurm_partition="PARTITION",
      slurm_job_name="EXAMPLE",
      tasks_per_node=1,
      cpus_per_task=8,
      mem="1GB",
  )
  job = run_on_slurm[slurm_config](1, b=2)
  result = job.result()  # block and get the result
```
