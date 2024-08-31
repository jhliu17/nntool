import os
import toml

from pathlib import Path
from dataclasses import dataclass
from .utils_module import get_output_path


@dataclass
class BaseExperimentConfig:
    # the output folder for the outputs
    output_folder: str = "outputs"

    # key for experiment name from the environment variable
    experiment_name_key: str = "EXP_NAME"

    # the path to the env.toml file
    env_toml_path: str = "env.toml"

    # append date time to the output path
    append_date_to_path: bool = True

    # random seed
    seed: int = 3407

    def __post_init__(self):
        # annotations
        self.experiment_name: str
        self.project_path: str
        self.output_path: str
        self.current_time: str
        self.env_config = self.prepare_env_toml_dict()

        self.experiment_name = self.prepare_experiment_name()
        self.project_path, self.output_path, self.current_time = (
            self.prepare_experiment_paths()
        )

        # custom post update for the derived class
        self.post_config_fields()

    def prepare_env_toml_dict(self):
        env_toml_path = Path(self.env_toml_path)
        if not env_toml_path.exists():
            raise FileNotFoundError(f"{env_toml_path} does not exist")

        with open(self.env_toml_path, "r") as f:
            config = toml.load(f)
        return config

    def prepare_experiment_name(self):
        return os.environ.get(self.experiment_name_key, "default")

    def prepare_experiment_paths(self):
        project_path = self.env_config["project"]["path"]

        output_path, current_time = get_output_path(
            output_path=f"{self.output_folder}/{self.experiment_name}",
            append_date=self.append_date_to_path,
        )
        output_path = f"{project_path}/{output_path}"
        return project_path, output_path, current_time

    def post_config_fields(self):
        """post configuration steps for the derived class"""
        pass
