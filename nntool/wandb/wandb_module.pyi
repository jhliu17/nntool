class WandbConfig:
    # project name in wandb
    project: str

    # wandb user name
    entity: str

    # wandb run name (leave blacnk to use default name)
    name: str

    # wandb run notes
    notes: str

    # log git hash
    log_git_hash: bool

    # log code
    log_code: bool

    # code root
    code_root: str

    # code file extensions
    code_ext: list[str]

    # wandb api key (toml file with [wandb] key field)
    api_key_config_file: str

def init_wandb(args: WandbConfig, run_config: dict) -> None: ...
