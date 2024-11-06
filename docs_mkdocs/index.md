---
hide:
  - navigation
---
# NNTool Documentation

NNTool is a package built on top of `submitit` designed to provide simple abstractions to submit a Python function to Slurm for machine learning research. Below is an example to run a Python function on a slurm cluster using specific configurations.

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

  result = job.result()  # block and get the result => 3
```

## Installation

NNTool is tested and supported on the following systems:

* Python 3.9–3.12
* Linux systems

```bash
pip install nntool-1.2.2-cp3xx-cp3xx-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

Please contact the developer at <junhaoliu17@gmail.com> to request pre-built wheels.
