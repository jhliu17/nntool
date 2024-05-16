class WandbConfig:
    project: str
    entity: str
    name: str
    notes: str
    log_git_hash: bool
    log_code: bool
    code_root: str
    code_ext: list[str]
    api_key_config_file: str

def init_wandb(args: WandbConfig, run_config: dict) -> None: ...
