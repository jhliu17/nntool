---
hide:
  - navigation
  - toc
---
# Welcome to 🚂 NNTool

Using NNTool to submit a Python function to Slurm and boost deep learning research.

## Get started with NNTool

```py
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

### Install NNTool

NNTool is tested and supported on the following systems:

* Python 3.9–3.12
* Linux systems

```bash title="Install NNTool with Python's pip package manager."
pip install nntool
```
