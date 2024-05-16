import os
import git
import wandb
import toml

from dataclasses import dataclass, field


@dataclass
class WandbConfig:
    # project name in wandb
    project: str = "YOUR_PROJECT_NAME"

    # wandb user name
    entity: str = "YOUR_ENTITY_NAME"

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

    # wandb api key (toml file with [wandb] key field)
    api_key_config_file: str = ""


def init_wandb(args: WandbConfig, run_config: dict):
    if "WANDB_API_KEY" in os.environ["WANDB_API_KEY"]:
        wandb.login(key=os.environ["WANDB_API_KEY"])
    elif args.api_key_config_file:
        with open(args.api_key_config_file, "r") as config_file:
            config_data = toml.load(config_file)
        wandb.login(key=config_data["wandb"]["key"])
    else:
        pass

    if args.log_git_hash:
        repo = git.Repo(search_parent_directories=True)
        git_hash = repo.head.object.hexsha
        args.notes = (
            f"{args.notes + ', ' if args.notes else ''}" + f"git hash: {git_hash}"
        )

    wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name if args.name else None,
        notes=args.notes,
        config=run_config,
    )
    wandb.run.log_code(
        root=args.code_root,
        include_fn=lambda path, root: any(
            [path.endswith(ext) for ext in args.code_ext]
        ),
    )
