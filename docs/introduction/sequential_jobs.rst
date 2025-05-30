Sequential Jobs with Slurm
##########################

This example shows how to submit a series of jobs with ``nntool``.

Configuration of SlurmFunction
==============================

The ``slurm_fn`` decorator converts a Python function into a SlurmFunction. The SlurmFunction can be used to submit jobs to the Slurm cluster. Each configuration of a SlurmFunction will create a new copy of the function. This is useful when you want to run the same function with different configurations.

.. code-block:: python
    :caption: A worker function with slurm settings

    import time
    from nntool.slurm import SlurmConfig, slurm_fn

    slurm_settings = SlurmConfig(
        mode="slurm",
        job_name="JOB_NAME",
        partition="PATITION",
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


.. important::

    A ``SlurmFunction`` executed on the Slurm cluster is non-blocking. The ``result()`` method is used to get the result of the job. The ``result()`` method will block until the job is finished.


Map array
=========

.. code-block:: python
    :caption: A worker function maps an array

    fn = work_fn[slurm_settings]

    job = fn(1, 2)
    result = job.result()
    print(result)
    assert result == 3

    jobs = fn.map_array([1, 2, 8, 9], [3, 4, 8, 9])
    results = [job.result() for job in jobs]
    print(results)
    assert results == [4, 6, 16, 18]


Dependency between jobs
=======================

.. code-block:: python
    :caption: A worker function runs sequentially

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

