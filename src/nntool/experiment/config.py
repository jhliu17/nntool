import os

from typing import Any, Dict
from pathlib import Path
from dataclasses import dataclass
from .utils import get_output_path, read_toml_file


@dataclass
class BaseExperimentConfig:
    """
    Configuration class for setting up an experiment.

    :param config_name: The name of the configuration.
    :param output_folder: The folder path where the outputs will be saved.
    :param experiment_name_key: Key for experiment name in the environment variable, default is 'EXP_NAME'.
    :param env_toml_path: Path to the `env.toml` file, default is 'env.toml'.
    :param append_date_to_path: If True, the current date and time will be appended to the output path, default is True.
    :param existing_output_path_ok: If True, the existing output path is ok to be reused, default is False.
    """

    # config name
    config_name: str

    # the output folder for the outputs
    output_folder: str

    # key for experiment name in the environment variable
    experiment_name_key: str = "EXP_NAME"

    # the path to the env.toml file
    env_toml_path: str = "env.toml"

    # append date time to the output path
    append_date_to_path: bool = True

    # exisiting output path is ok
    existing_output_path_ok: bool = False

    def __post_init__(self):
        # annotations
        self.experiment_name: str
        self.project_path: str
        self.output_path: str
        self.current_time: str
        self.env_toml: Dict[str, Any] = self.__prepare_env_toml_dict()

        self.experiment_name = self.__prepare_experiment_name()
        self.project_path, self.output_path, self.current_time = self.__prepare_experiment_paths()

        # custom post update for the derived class
        self.set_up_stateful_fields()

    def __prepare_env_toml_dict(self):
        env_toml_path = Path(self.env_toml_path)
        if not env_toml_path.exists():
            raise FileNotFoundError(f"{env_toml_path} does not exist")

        config = read_toml_file(env_toml_path)
        return config

    def __prepare_experiment_name(self):
        return os.environ.get(self.experiment_name_key, "default")

    def __prepare_experiment_paths(self):
        project_path = self.env_toml["project"]["path"]

        output_path, current_time = get_output_path(
            output_path=os.path.join(self.output_folder, self.config_name, self.experiment_name),
            append_date=self.append_date_to_path,
            cache_into_env=False,
        )
        output_path = f"{project_path}/{output_path}"
        return project_path, output_path, current_time

    def get_output_path(self) -> str:
        """Return the output path prepared for the experiment.

        :return: output path for the experiment
        """
        return self.output_path

    def get_current_time(self) -> str:
        """Return the current time for the experiment.

        :return: current time for the experiment
        """
        return self.current_time

    def set_up_stateful_fields(self):
        """
        Post configuration steps for stateful fields such as `output_path` in the derived class.
        This method should be overridden in the derived class.
        """
        pass

    def start(self):
        """
        Start the experimen. This will
        - cache `NNTOOL_OUTPUT_PATH` and `NNTOOL_OUTPUT_PATH_DATE` into environment variables, which means the later launched processes would inherit these variables.
        - create the output path if it does not exist.
        """
        os.environ["NNTOOL_OUTPUT_PATH"] = self.get_output_path()
        os.environ["NNTOOL_OUTPUT_PATH_DATE"] = self.get_current_time()

        # create the output path
        output_path = Path(self.get_output_path())
        output_path.mkdir(parents=True, exist_ok=self.existing_output_path_ok)
