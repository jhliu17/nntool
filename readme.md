# 🚂 NNTool

[![Pytest](https://github.com/jhliu17/nntool/actions/workflows/pytest.yml/badge.svg)](https://github.com/jhliu17/nntool/actions/workflows/pytest.yml) [![Documentation](https://github.com/jhliu17/nntool/actions/workflows/docs.yml/badge.svg)](https://github.com/jhliu17/nntool/actions/workflows/docs.yml)

`nntool` is a package designed to provide seamless Python function execution on Slurm for machine learning research, with useful utilities for experiment tracking and management.

## Example

```python

   from nntool.slurm import SlurmConfig, slurm_fn

   @slurm_fn
   def run_on_slurm(a, b):
      return a + b

   slurm_config = SlurmConfig(
      mode="slurm",
      partition="PARTITION",
      job_name="EXAMPLE",
      tasks_per_node=1,
      cpus_per_task=8,
      mem="1GB",
   )

   job = run_on_slurm[slurm_config](1, b=2) # job is submitted to slurm
   result = job.result() # block and get the result => 3
```

## Installation

nntool is tested and supported on the following systems:

* Python 3.10-3.13
* Linux systems with Slurm installed

Install nntool via pip

```bash
   pip install nntool
```

## Development

### Development Installation

```bash
pip install -e ".[dev]"
```

### Testing

```bash
pytest
```

### Build Wheel

```bash
uv build
```
