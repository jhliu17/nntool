.. nntool documentation master file, created by
   sphinx-quickstart on Wed Nov  6 11:28:43 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NNTool Documentation
====================

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

* Python 3.9–3.12
* Linux systems

.. code-block:: sh

   pip install nntool-1.2.2-cp3xx.whl


Please contact the developer at `junhaoliu17@gmail.com <junhaoliu17@gmail.com>`_ to request pre-built wheels.


.. toctree::
   :caption: Examples
   :maxdepth: 2
   :hidden:

   examples/sequential_jobs
   examples/distributed_training


.. toctree::
   :caption: API reference
   :maxdepth: 2
   :hidden:

   nntool.slurm<documentations/slurm>
   nntool.experiment<documentations/experiment>
   nntool.wandb<documentations/wandb>
   nntool.utils<documentations/utils>
