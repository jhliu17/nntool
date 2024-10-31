from dataclasses import dataclass, field
from nntool.slurm import SlurmConfig
from nntool.experiment import BaseExperimentConfig


def test_output_path(tmp_path):
    base_slurm = SlurmConfig(
        mode="slurm",
        slurm_partition="zhanglab.p",
        slurm_job_name="dna_llm",
        tasks_per_node=1,
        cpus_per_task=8,
        gpus_per_node=4,
        processes_per_task=4,
        mem="192GB",
        node_list="laniakea",
        pack_code=True,
        use_packed_code=True,
        exclude_code_folders=[
            "wandb",
            "outputs",
            "datasets",
            "encode",
            "caches",
        ],
        use_distributed_env=True,
        distributed_launch_command="accelerate launch --config_file configs/accelerate/ddp_gpu4.yaml --num_processes {num_processes} --num_machines {num_machines} --machine_rank {machine_rank} --main_process_ip {main_process_ip} --main_process_port {main_process_port} -m scripts.multimodal",
    )

    @dataclass
    class ExperimentConfig(BaseExperimentConfig):
        slurm: SlurmConfig = field(default_factory=lambda: SlurmConfig)

        def set_up_stateful_fields(self):
            self.slurm = self.slurm.set_output_path(self.output_path)

    experiments = dict(
        base=ExperimentConfig(
            config_name="base",
            output_folder=str(tmp_path),
            slurm=base_slurm,
            append_date_to_path=True,
            env_toml_path="tests/env.toml",
        ),
        lite=ExperimentConfig(
            config_name="lite",
            output_folder=str(tmp_path),
            slurm=base_slurm,
            append_date_to_path=True,
            env_toml_path="tests/env.toml",
        ),
    )

    args = experiments["base"]
    assert (
        args.slurm.slurm_output_folder
        == f"{args.project_path}/{tmp_path}/base/{args.experiment_name}/{args.current_time}/slurm"
    )

    args = experiments["lite"]
    assert (
        args.slurm.slurm_output_folder
        == f"{args.project_path}/{tmp_path}/lite/{args.experiment_name}/{args.current_time}/slurm"
    )
