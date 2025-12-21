.. nntool documentation master file, created by
   sphinx-quickstart on Wed Nov  6 11:28:43 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NNTool
======

NNTool is a package built on top of ``submitit`` designed to provide simple abstractions to conduct experiments on Slurm for machine learning research. Below is an example to run a Python function on a slurm cluster using specific configurations.

.. code-block:: python

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
   job = run_on_slurm[slurm_config](1, b=2)

   result = job.result() # block and get the result => 3


Installation
------------

NNTool is tested and supported on the following systems:

* Python 3.10-3.13
* Linux systems with Slurm installed

.. code-block:: bash
   pip install nntool


Documentation
-------------

.. toctree::
   :caption: Getting started

   tutorials/sequential_jobs
   tutorials/distributed_training
   tutorials/sharp_bits


.. toctree::
   :caption: API reference
   :maxdepth: 2
   :hidden:

   api/nntool.slurm
   api/nntool.wandb
   api/nntool.experiment
