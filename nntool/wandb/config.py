import os
import git
import wandb
import tomli
import warnings

from dataclasses import dataclass, field
from typing import Union
from wandb.sdk.lib.disabled import RunDisabled
from wandb.sdk.wandb_run import Run


@dataclass
class WandbConfig:
    """
    Configuration class for Weights and Biases (wandb) integration.

    :param api_key_config_file: The file path to the configuration file containing the wandb API key.
                                The file should be a toml file with a `[wandb]` section. Default is an empty string.

    :param project: The name of the project in wandb. Default is an empty string.

    :param entity: The wandb user or team name. Default is an empty string.

    :param name: The name of the wandb run. Leave blank to use the default run name. Default is an empty string.

    :param notes: Notes or comments for the wandb run. Default is an empty string.

    :param log_git_hash: Whether to log the current Git hash. Default is True.

    :param log_code: Whether to log the current codebase. Default is True.

    :param code_root: The root directory of the codebase to be logged. Default is the current directory (`.`).

    :param code_ext: A list of file extensions for the code files to be logged. Default includes `.py` and `.sh`.
    """

    # wandb api key (toml file with [wandb] key field)
    api_key_config_file: str = ""

    # project name in wandb
    project: str = ""

    # wandb user name
    entity: str = ""

    # wandb run name (leave blacnk to use default name)
    name: str = ""

    # wandb run notes
    notes: str = ""

    # log git hash
    log_git_hash: bool = True

    # log code
    log_code: bool = True

    # code root
    code_root: str = "."

    # code file extensions
    code_ext: list[str] = field(default_factory=lambda: [".py", ".sh"])


def is_wandb_enabled():
    return wandb.run is not None


def init_wandb(args: WandbConfig, run_config: dict) -> Union[Run, RunDisabled, None]:
    """initialize wandb and log the configuration

    :param args: WandbConfig object
    :param run_config: configuration dictionary to be logged
    :return: wandb run object
    """
    project, entity = args.project, args.entity
    if "WANDB_API_KEY" in os.environ:
        warnings.warn("WANDB_API_KEY is found in environment variables. Using it.")
        wandb.login(key=os.environ["WANDB_API_KEY"])
    elif args.api_key_config_file:
        with open(args.api_key_config_file, "rb") as config_file:
            config_data = tomli.load(config_file)
        wandb.login(key=config_data["wandb"]["key"])
        project = config_data["wandb"].get("project", args.project)
        entity = config_data["wandb"].get("entity", args.entity)
    else:
        warnings.warn("WANDB_API_KEY is not found in environment variables or the local key file.")
        pass

    if args.log_git_hash:
        repo = git.Repo(search_parent_directories=True)
        git_hash = repo.head.object.hexsha
        args.notes = f"{args.notes + ', ' if args.notes else ''}" + f"git hash: {git_hash}"

    wandb_run = wandb.init(
        project=project,
        entity=entity,
        name=args.name if args.name else None,
        notes=args.notes,
        config=run_config,
    )
    if not is_wandb_enabled():
        warnings.warn("wandb is not enabled after intialization. Please check `wandb enabled`.")

    wandb_run.log_code(
        root=args.code_root,
        include_fn=lambda path, root: any([path.endswith(ext) for ext in args.code_ext]),
    )
    return wandb_run
