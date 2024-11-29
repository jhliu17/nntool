#########################
Distributed training jobs
#########################

Example
=======

Here is an example of using ``nntool`` to launch a distributed training with ``accelerate``.

.. code-block:: python

    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from nntool.slurm import slurm_fn, SlurmConfig

    @slurm_fn
    def main(config: SlurmConfig):
        accelerator = Accelerator()

        # TRAINING
        ...


Important variables
===================

To launch a distributed job, it is necessary to set up the ``use_distributed_env`` and ``distributed_lanch_command`` in the ``SlurmConfig`` function. The ``distributed_launch_command`` is a command that is used to launch the distributed job. There are several arguments exposed by the ``nntool`` which are useful to set up the distributed job. The arguments are as follows:

- ``num_processes``: int
- ``num_machines``: int
- ``machine_rank``: int
- ``main_process_ip``: str
- ``main_process_port``: int

.. code-block:: python

    if __name__ == "__main__":
        slurm_config = SlurmConfig(
            mode="slurm",
            partition="PARITITION",
            job_name="JOB_NAME",
            tasks_per_node=1,
            cpus_per_task=8,
            gpus_per_node=4,
            processes_per_task=4,
            mem="192GB",
            node_list="NODE_LIST",
            use_distributed_env=True,
            distributed_launch_command="accelerate launch --config_file CONFIG_FILE --num_processes {num_processes} --num_machines {num_machines} --machine_rank {machine_rank} --main_process_ip {main_process_ip} --main_process_port {main_process_port} main.py",
        )
        main[slurm_config](config)

