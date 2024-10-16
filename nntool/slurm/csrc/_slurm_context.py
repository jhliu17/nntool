import shlex

from submitit import SlurmExecutor
from ..config import SlurmConfig
from ..task import Task


class patch_submitit_distributed_command_str:
    def __init__(self, config: SlurmConfig, task: Task):
        self.config = config
        self.task = task
        self.is_patched = False
        self.previous_submitit_command_str = None

    def __enter__(self):
        # monkey patch the submitit command to set up distributed env
        # in distributed training, if two jobs are launched in the same node, the second job will fail
        # but directly use `sbatch`` to submit the second job without any issues
        if self.slurm_config.mode == "slurm":
            task_command = self.task.command()

            def _submitit_command_str(self) -> str:
                return " ".join(
                    [
                        self.python,
                        "-u -m submitit.core._submit",
                        shlex.quote(str(self.folder)),
                        "\n".join(
                            [
                                "\n",
                                "# nntool command",
                                "export NNTOOL_SLURM_HAS_BEEN_SET_UP=1",
                                f"source {shlex.quote(str(self.folder))}/nntool_distributed_env.sh",
                                f"{task_command}",
                            ]
                        ),
                    ]
                )

            self.previous_submitit_command_str = SlurmExecutor._submitit_command_str
            SlurmExecutor._submitit_command_str = property(_submitit_command_str)
            self.is_patched = True

    def __exit__(self, exc_type, exc_value, traceback):
        if self.is_patched:
            SlurmExecutor._submitit_command_str = self.previous_submitit_command_str
