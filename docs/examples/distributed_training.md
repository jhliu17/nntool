```py title="main.py"
from accelerate import Accelerator
from accelerate.utils import set_seed
from nntool.slurm import slurm_fn, SlurmConfig

@slurm_fn
def main(config: SlurmConfig):
    accelerator = Accelerator()

    # TRAINING
    ...


if __name__ == "__main__":
    slurm_config = SlurmConfig(
        mode="slurm",
        slurm_partition="PARITITION",
        slurm_job_name="JOB_NAME",
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
```
