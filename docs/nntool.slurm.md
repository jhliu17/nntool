::: nntool.slurm.function.SlurmFunction
    options:
        members:
        - __init__
        - is_configured
        - is_distributed
        - slurm_has_been_set_up
        - configure
        - submit
        - map_array
        - on_condition
        - afterok
        - afterany
        - afternotok

# Configure Slurm Function

::: nntool.slurm.args
    options:
        members:
            - SlurmConfig
        show_source: true

::: nntool.slurm.wrap
    options:
      members:
        - slurm_fn
