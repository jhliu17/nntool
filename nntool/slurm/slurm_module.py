import os
import sys
import shlex
import submitit

from copy import deepcopy
from functools import partial
from submitit import Job, SlurmExecutor
from warnings import warn
from typing import Any, Callable, Literal, Tuple, Type, Union, Dict, List

from ..parser import parse_from_cli
from .args import SlurmConfig
from .task import (
    PyTorchDistributedTask,
    pack_code_files,
    include_code_files,
    exclude_code_folders,
)


class SlurmFunction:
    def __init__(
        self,
        submit_fn: Callable[..., Any] = None,
        default_submit_fn_args: Union[Tuple[Any], None] = None,
        default_submit_fn_kwargs: Union[Dict[str, Any], None] = None,
    ):
        """A slurm function for the slurm job, which can be used for distributed or non-distributed job (controlled by `use_distributed_env` in the slurm dataclass).

        :param submit_fn: function to be submitted to Slurm, defaults to None
        :param default_submit_fn_args: default args for submit_fn, defaults to ()
        :param default_submit_fn_kwargs: default known word args for submit_fn, defaults to {}
        :return: the wrapped submit function with configured slurm paramters
        """
        self.submit_fn: Callable[..., Any] = submit_fn
        self.default_submit_fn_args: Tuple[Any] = (
            tuple() if default_submit_fn_args is None else default_submit_fn_args
        )
        self.default_submit_fn_kwargs: Dict[str, Any] = (
            dict() if default_submit_fn_kwargs is None else default_submit_fn_kwargs
        )
        self.__doc__ = self.submit_fn.__doc__

        # slurm funcion is instantiated after calling `instantiate`
        self._instantiated = False
        self.executor = None  # to be set up by `get_executor`

        # annotations here, will be set up after instantiation
        self.slurm_config: SlurmConfig
        self.slurm_params_kwargs: Dict[str, str]
        self.slurm_submit_kwargs: Dict[str, str]
        self.slurm_task_kwargs: Dict[str, str]
        self.system_argv: Union[List[str], None]
        self.pack_code_include_fn: Callable[[str, str], bool]
        self.pack_code_exclude_fn: Callable[[str, str], bool]

    def is_instantiated(self):
        """Whether the slurm function has been set up.

        :return: True if the slurm function has been set up, False otherwise
        """
        return self.submit_fn is not None and self._instantiated

    def is_distributed(self):
        """Whether the slurm function is distributed.

        :return: True if the slurm function is distributed, False otherwise
        """
        return self.slurm_config.use_distributed_env

    def get_executor(
        self,
    ):
        if self.executor is not None:
            return self.executor

        slurm_config = self.slurm_config
        slurm_parameters_kwargs = self.slurm_params_kwargs
        slurm_submission_kwargs = self.slurm_submit_kwargs

        # select the cluster type, which is based on the submitit library
        # here we add a special mode called `exec` for running the job in the local machine
        # which is equivalent to the `debug` mode in the submitit library
        cluster_dispatch = {"slurm": None, "exec": "debug", "debug": "debug", "local": "local"}
        executor = submitit.AutoExecutor(
            folder=slurm_config.slurm_output_folder,
            cluster=cluster_dispatch.get(slurm_config.mode, slurm_config.mode),
        )

        # set additional parameters
        slurm_additional_parameters = {}
        if slurm_config.node_list:
            slurm_additional_parameters["nodelist"] = slurm_config.node_list
        if slurm_config.node_list_exclude:
            slurm_additional_parameters["exclude"] = slurm_config.node_list_exclude
        if slurm_config.mem:
            slurm_additional_parameters["mem"] = slurm_config.mem
        slurm_additional_parameters.update(slurm_parameters_kwargs)

        # set slurm parameters
        executor.update_parameters(
            name=slurm_config.slurm_job_name,
            slurm_partition=slurm_config.slurm_partition,
            nodes=slurm_config.num_of_node,
            slurm_tasks_per_node=slurm_config.tasks_per_node,
            slurm_cpus_per_task=slurm_config.cpus_per_task,
            slurm_gpus_per_node=(
                slurm_config.gpus_per_task * slurm_config.tasks_per_node
                if slurm_config.gpus_per_node is None
                else slurm_config.gpus_per_node
            ),  # gpu cannot be assigned in the task level
            timeout_min=slurm_config.timeout_min,
            slurm_additional_parameters=slurm_additional_parameters,
            **slurm_submission_kwargs,
        )

        # set executor
        self.executor = executor
        return executor

    @staticmethod
    def slurm_has_been_set_up() -> bool:
        """NNTOOL_SLURM_HAS_BEEN_SET_UP is a special environment variable to indicate that the slurm has been set up.

        :return: bool
        """
        # check whether slurm has been set up
        has_been_set_up = False
        if os.environ.get("NNTOOL_SLURM_HAS_BEEN_SET_UP") is not None:
            has_been_set_up = True
        return has_been_set_up

    def _mark_slurm_has_been_set_up(self):
        os.environ["NNTOOL_SLURM_HAS_BEEN_SET_UP"] = "1"

    def _update_slurm_kwargs(
        self,
        slurm_params_kwargs: Dict[str, str] = {},
        slurm_submit_kwargs: Dict[str, str] = {},
        slurm_task_kwargs: Dict[str, str] = {},
    ):
        """update the slurm configuration for the slurm function. By default, the slurm parameters, slurm submission parameters, and slurm task parameters are updated. The slurm parameters are updated by the slurm configuration, while the slurm submission parameters and slurm task parameters would override them by the given arguments.

        :param slurm_params_kwargs: extra settings, defaults to {}
        :param slurm_submit_kwargs: extra settings, defaults to {}
        :param slurm_task_kwargs: extra settings, defaults to {}
        """
        if slurm_params_kwargs:
            self.slurm_params_kwargs.update(slurm_params_kwargs)
        if slurm_submit_kwargs:
            self.slurm_submit_kwargs.update(slurm_submit_kwargs)
        if slurm_task_kwargs:
            self.slurm_task_kwargs.update(slurm_task_kwargs)

    def instantiate(
        self,
        slurm_config: SlurmConfig,
        slurm_params_kwargs: Union[Dict[str, str], None] = None,
        slurm_submit_kwargs: Union[Dict[str, str], None] = None,
        slurm_task_kwargs: Union[Dict[str, str], None] = None,
        system_argv: Union[List[str], None] = None,
        pack_code_include_fn: Callable[[str, str], bool] = None,
        pack_code_exclude_fn: Callable[[str, str], bool] = None,
    ) -> "SlurmFunction":
        """Update the slurm configuration for the slurm function. A slurm function for the slurm job, which can be used for distributed or non-distributed job (controlled by `use_distributed_env` in the slurm dataclass).

        #### Exported Distributed Enviroment Variables
        1. NNTOOL_SLURM_HAS_BEEN_SET_UP is a special environment variable to indicate that the slurm has been set up.
        2. After the set up, the distributed job will be launched and the following variables are exported:         num_processes: int, num_machines: int, machine_rank: int, main_process_ip: str, main_process_port: int.

        :param slurm_config: SlurmConfig, the slurm configuration dataclass, defaults to None
        :param slurm_params_kwargs: extra slurm arguments for the slurm configuration, defaults to {}
        :param slurm_submit_kwargs: extra slurm arguments for `srun` or `sbatch`, defaults to {}
        :param slurm_task_kwargs: extra arguments for the setting of distributed task, defaults to {}
        :param system_argv: the system arguments for the second launch in the distributed task (by default it will use the current system arguments `sys.argv[1:]`), defaults to None
        :return: the wrapped submit function with configured slurm paramters
        """
        slurm_fn = SlurmFunction(
            submit_fn=self.submit_fn,
            default_submit_fn_args=self.default_submit_fn_args,
            default_submit_fn_kwargs=self.default_submit_fn_kwargs,
        )

        slurm_fn.slurm_config = slurm_config
        slurm_fn.slurm_params_kwargs = (
            {} if slurm_params_kwargs is None else deepcopy(slurm_params_kwargs)
        )
        slurm_fn.slurm_submit_kwargs = (
            {} if slurm_submit_kwargs is None else deepcopy(slurm_submit_kwargs)
        )
        slurm_fn.slurm_task_kwargs = (
            {} if slurm_task_kwargs is None else deepcopy(slurm_task_kwargs)
        )
        slurm_fn.system_argv = system_argv

        slurm_fn._update_slurm_kwargs(
            slurm_fn.slurm_config.slurm_params_kwargs,  # make sure the same parameters are controlled by the config
            slurm_fn.slurm_config.slurm_submit_kwargs,
            slurm_fn.slurm_config.slurm_task_kwargs,
        )

        slurm_fn.pack_code_include_fn = partial(
            include_code_files,
            code_ext=slurm_fn.slurm_config.code_file_suffixes,
        )
        slurm_fn.pack_code_exclude_fn = partial(
            exclude_code_folders,
            code_folders=slurm_fn.slurm_config.exclude_code_folders,
        )

        if pack_code_include_fn is not None:
            slurm_fn.pack_code_include_fn = pack_code_include_fn

        if pack_code_exclude_fn is not None:
            slurm_fn.pack_code_exclude_fn = pack_code_exclude_fn

        # mark instantiated
        slurm_fn._instantiated = True
        return slurm_fn

    def __getitem__(
        self, slurm_configs: Union[Dict[str, Any], Tuple[Any], Any]
    ) -> "SlurmFunction":
        """Instantiate the slurm configuration for the slurm function. A slurm function for the slurm job, which can be used for distributed or non-distributed job (controlled by `use_distributed_env` in the slurm dataclass).

        #### Exported Distributed Enviroment Variables
        1. NNTOOL_SLURM_HAS_BEEN_SET_UP is a special environment variable to indicate that the slurm has been set up.
        2. After the set up, the distributed job will be launched and the following variables are exported:         num_processes: int, num_machines: int, machine_rank: int, main_process_ip: str, main_process_port: int.

        :param slurm_config: SlurmConfig, the slurm configuration dataclass, defaults to None
        :param slurm_params_kwargs: extra slurm arguments for the slurm configuration, defaults to {}
        :param slurm_submit_kwargs: extra slurm arguments for `srun` or `sbatch`, defaults to {}
        :param slurm_task_kwargs: extra arguments for the setting of distributed task, defaults to {}
        :param system_argv: the system arguments for the second launch in the distributed task (by default it will use the current system arguments `sys.argv[1:]`), defaults to None
        :return: the wrapped submit function with configured slurm paramters
        """
        if isinstance(slurm_configs, dict):
            return self.instantiate(**slurm_configs)
        elif isinstance(slurm_configs, (list, tuple)):
            return self.instantiate(*slurm_configs)
        else:
            # will try to pass the slurm_configs as the first argument
            return self.instantiate(slurm_configs)

    def _before_submit(self, *args, **kwargs):
        """The hook function before submitting the job. It will pack the code and scripts to the slurm output folder if the `pack_code` is set to True in the slurm configuration. Only work before the first submit.

        :raises Exception: if the slurm function is not integrated
        """
        if self.slurm_has_been_set_up():
            return

        if not self.is_instantiated():
            raise Exception("Slurm function should be instantiated before calling.")

        # pack the code and scripts to the slurm output folder
        if self.slurm_config.pack_code:
            target_code_root = pack_code_files(
                self.slurm_config.code_root,
                self.slurm_config.slurm_output_folder,
                include_fn=self.pack_code_include_fn,
                exclude_fn=self.pack_code_exclude_fn,
            )

            # set sbatch command to change directory
            if self.slurm_config.use_packed_code:
                self.slurm_params_kwargs.update({"chdir": target_code_root})

    def _after_submit(
        self,
        submit_results: Union[Job, Any] = None,
        *args,
        **kwargs,
    ):
        # get result to run program other than slurm mode
        if isinstance(submit_results, Job):
            if self.slurm_config.mode != "slurm":
                submit_results.results()

    def __call__(self, *submit_fn_args, **submit_fn_kwargs) -> Union[Job, Any]:
        """Run the submit_fn with the given arguments and keyword arguments. The function is non-blocking in the mode of `slurm`, while other modes cause blocking. If there is no given arguments or keyword arguments, the default arguments and keyword arguments will be used.

        :raises Exception: if the submit_fn is not set up
        :return: Slurm Job or the return value of the submit_fn
        """
        self._before_submit()
        submit_strategy = self._dispatch_submit_strategy("submit")
        submit_results = submit_strategy(*submit_fn_args, **submit_fn_kwargs)
        self._after_submit(submit_results)
        return submit_results

    def _dispatch_submit_strategy(
        self,
        submit_mode: Literal["submit", "map_array"] = "submit",
        *submit_fn_args,
        **submit_fn_kwargs,
    ) -> Callable[..., Union[Job, Any]]:
        if submit_mode == "submit":
            if self.is_distributed():
                return partial(self._distributed_submit, submit_mode="submit")
            else:
                return partial(self._submit, submit_mode="submit")
        elif submit_mode == "map_array":
            if self.is_distributed():
                raise Exception("Distributed job does not support `map_array` mode.")
            else:
                return partial(self._submit, submit_mode="map_array")
        else:
            raise Exception(f"Invalid submit mode: {submit_mode}")

    def submit(self, *submit_fn_args, **submit_fn_kwargs) -> Union[Job, Any]:
        """An alias function to __call__.

        :raises Exception: if the submit_fn is not set up
        :return: Slurm Job or the return value of the submit_fn
        """
        return self(*submit_fn_args, **submit_fn_kwargs)

    def map_array(self, *submit_fn_args, **submit_fn_kwargs) -> List[Job]:
        """Run the submit_fn with the given arguments and keyword arguments. The function is non-blocking in the mode of `slurm`, while other modes cause blocking. If there is no given arguments or keyword arguments, the default arguments and keyword arguments will be used.

        :raises Exception: if the submit_fn is not set up
        :return: Slurm Job or the return value of the submit_fn
        """
        self._before_submit()
        submit_strategy = self._dispatch_submit_strategy("map_array")
        submit_results = submit_strategy(*submit_fn_args, **submit_fn_kwargs)
        self._after_submit(submit_results)
        return submit_results

    def on_condition(
        self,
        jobs: Union[Job, List[Job], Tuple[Job]],
        condition: Literal["afterany", "afterok", "afternotok"] = "afterok",
    ):
        """Mark this job should be executed after the provided slurm jobs have been done. This function allows combining different conditions by multiple calling.

        :param jobs: dependent jobs
        :param condition: run condition, defaults to "afterok"
        :return: self
        """
        if not isinstance(jobs, (list, tuple)):
            jobs = [jobs]

        previous_conditions = self.slurm_params_kwargs.get("dependency", "")
        append_condition = f"{condition}:{':'.join([job.job_id for job in jobs])}"
        self.slurm_params_kwargs.update(
            {
                "dependency": (
                    f"{previous_conditions}:{append_condition}"
                    if previous_conditions
                    else append_condition
                )
            }
        )
        return self

    def afterok(self, *jobs: Tuple[Job]):
        return self.on_condition(jobs, "afterok")

    def afterany(self, *jobs: Tuple[Job]):
        return self.on_condition(jobs, "afterany")

    def afternotok(self, *jobs: Tuple[Job]):
        return self.on_condition(jobs, "afternotok")

    def _submit(
        self,
        submit_mode: Literal["submit", "map_array"] = "submit",
        *submit_fn_args,
        **submit_fn_kwargs,
    ) -> Job:
        submit_fn_args = (
            self.default_submit_fn_args if not submit_fn_args else submit_fn_args
        )
        submit_fn_kwargs = (
            self.default_submit_fn_kwargs if not submit_fn_kwargs else submit_fn_kwargs
        )

        executor = self.get_executor()
        self._mark_slurm_has_been_set_up()
        job = getattr(executor, submit_mode)(
            self.submit_fn, *submit_fn_args, **submit_fn_kwargs
        )
        return job

    def _distributed_submit(
        self,
        submit_mode: Literal["submit", "map_array"] = "submit",
        *submit_fn_args,
        **submit_fn_kwargs,
    ) -> Union[Job, Any]:
        submit_fn_args = (
            self.default_submit_fn_args if not submit_fn_args else submit_fn_args
        )
        submit_fn_kwargs = (
            self.default_submit_fn_kwargs if not submit_fn_kwargs else submit_fn_kwargs
        )

        if not self.slurm_has_been_set_up():
            # prepare distributed env for the second launch
            task = PyTorchDistributedTask(
                self.slurm_config.distributed_launch_command,
                (
                    self.system_argv
                    if self.system_argv is not None
                    else list(sys.argv[1:])
                ),
                self.slurm_config,
                verbose=True,
                **self.slurm_task_kwargs,
            )

            # monkey patch the submitit command to set up distributed env
            # in distributed training, if two jobs are launched in the same node, the second job will fail
            # directly use sbatch to submit jobs fixed the issue
            if self.slurm_config.mode == "slurm":

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
                                    f"{task.command()}",
                                ]
                            ),
                        ]
                    )

                SlurmExecutor._submitit_command_str = property(_submitit_command_str)

            executor = self.get_executor()
            self._mark_slurm_has_been_set_up()
            job = getattr(executor, submit_mode)(task)
            return job
        else:
            return self.submit_fn(*submit_fn_args, **submit_fn_kwargs)


def slurm_launcher(
    ArgsType: Type[Any],
    parser: Union[str, Callable] = "tyro",
    slurm_key: str = "slurm",
    slurm_params_kwargs: dict = {},
    slurm_submit_kwargs: dict = {},
    slurm_task_kwargs: dict = {},
    *extra_args,
    **extra_kwargs,
) -> Callable[[Callable[..., Any]], SlurmFunction]:
    """A slurm launcher decorator for distributed or non-distributed job (controlled by `use_distributed_env` in slurm field). This decorator should be used as the program entry. The decorated function is non-blocking in the mode of `slurm`, while other modes cause blocking.

    #### Exported Distributed Enviroment Variables
    1. NNTOOL_SLURM_HAS_BEEN_SET_UP is a special environment variable to indicate that the slurm has been set up.
    2. After the set up, the distributed job will be launched and the following variables are exported:         num_processes: int, num_machines: int, machine_rank: int, main_process_ip: str, main_process_port: int.

    :param ArgsType: the experiment arguments type, which should be a dataclass (it
                     mush have a slurm field defined by `slurm_key`)
    :param slurm_key: the key of the slurm field in the ArgsType, defaults to "slurm"
    :param parser: the parser for the arguments, defaults to "tyro"
    :param slurm_config: SlurmConfig, the slurm configuration dataclass
    :param slurm_params_kwargs: extra slurm arguments for the slurm configuration, defaults to {}
    :param slurm_submit_kwargs: extra slurm arguments for `srun` or `sbatch`, defaults to {}
    :param slurm_task_kwargs: extra arguments for the setting of distributed task, defaults to {}
    :param extra_args: extra arguments for the parser
    :param extra_kwargs: extra keyword arguments for the parser
    :return: decorator function with main entry
    """
    argv = list(sys.argv[1:])
    args = parse_from_cli(ArgsType, parser, *extra_args, **extra_kwargs)

    # check if args have slurm field
    if not hasattr(args, slurm_key):
        raise ValueError(
            f"ArgsType should have a field named `{slurm_key}` to use `slurm_launcher` decorator."
        )
    slurm_config: SlurmConfig = getattr(args, slurm_key)

    def decorator(
        submit_fn: Callable[..., Any],
    ) -> SlurmFunction:
        return SlurmFunction(
            submit_fn=submit_fn,
            default_submit_fn_args=(args,),
        ).instantiate(
            slurm_config,
            slurm_params_kwargs,
            slurm_submit_kwargs,
            slurm_task_kwargs,
            system_argv=argv,
        )

    return decorator


def slurm_distributed_launcher(
    ArgsType: Type[Any],
    parser: Union[str, Callable] = "tyro",
    slurm_key: str = "slurm",
    slurm_params_kwargs: dict = {},
    slurm_submit_kwargs: dict = {},
    slurm_task_kwargs: dict = {},
    *extra_args,
    **extra_kwargs,
) -> Callable[[Callable[..., Any]], SlurmFunction]:
    """A slurm launcher decorator for the distributed job. This decorator should be used for the distributed job only and as the program entry. The decorated function is non-blocking in the mode of `slurm`, while other modes cause blocking.

    #### Exported Distributed Enviroment Variables
    1. NNTOOL_SLURM_HAS_BEEN_SET_UP is a special environment variable to indicate that the slurm has been set up.
    2. After the set up, the distributed job will be launched and the following variables are exported:         num_processes: int, num_machines: int, machine_rank: int, main_process_ip: str, main_process_port: int.

    :param ArgsType: the experiment arguments type, which should be a dataclass (it
                     mush have a slurm field defined by `slurm_key`)
    :param slurm_key: the key of the slurm field in the ArgsType, defaults to "slurm"
    :param parser: the parser for the arguments, defaults to "tyro"
    :param slurm_config: SlurmConfig, the slurm configuration dataclass
    :param slurm_params_kwargs: extra slurm arguments for the slurm configuration, defaults to {}
    :param slurm_submit_kwargs: extra slurm arguments for `srun` or `sbatch`, defaults to {}
    :param slurm_task_kwargs: extra arguments for the setting of distributed task, defaults to {}
    :param extra_args: extra arguments for the parser
    :param extra_kwargs: extra keyword arguments for the parser
    :return: decorator function with main entry
    """
    warn(
        "`slurm_distributed_launcher` has been deprecated. Please use `slurm_launcher` instead, which supports both distributed or non-distributed job (controlled by `use_distributed_env` in slurm field).",
        DeprecationWarning,
        stacklevel=2,
    )
    argv = list(sys.argv[1:])
    args = parse_from_cli(ArgsType, parser, *extra_args, **extra_kwargs)

    # check if args have slurm field
    if not hasattr(args, slurm_key):
        raise ValueError(
            f"ArgsType should have a field named `{slurm_key}` to use `slurm_launcher` decorator."
        )
    slurm_config: SlurmConfig = getattr(args, slurm_key)

    def decorator(
        submit_fn: Callable[..., Any],
    ) -> SlurmFunction:
        return SlurmFunction(
            submit_fn=submit_fn,
            default_submit_fn_args=(args,),
        ).instantiate(
            slurm_config,
            slurm_params_kwargs,
            slurm_submit_kwargs,
            slurm_task_kwargs,
            system_argv=argv,
        )

    return decorator


def slurm_function(
    submit_fn: Callable,
):
    """A decorator to annoate a function to be run in slurm. The function decorated by this decorator should be launched in the way below.
    ```
    @slurm_function
    def run_in_slurm(*args, **kwargs):
        pass

    job = run_in_slurm(slurm_config)(*args, **kwargs)
    ```
    The decorated function `submit_fn` is non-blocking now. To block and get the return value, you can call `job.result()`.
    """

    def wrapper(
        slurm_config: SlurmConfig,
        slurm_params_kwargs: Dict[str, Any] = {},
        slurm_submit_kwargs: Dict[str, Any] = {},
        slurm_task_kwargs: Dict[str, Any] = {},
        system_argv: Union[List[str], None] = None,
    ) -> SlurmFunction:
        """Update the slurm configuration for the slurm function.

        #### Exported Distributed Enviroment Variables
        1. NNTOOL_SLURM_HAS_BEEN_SET_UP is a special environment variable to indicate that the slurm has been set up.
        2. After the set up, the distributed job will be launched and the following variables are exported:         num_processes: int, num_machines: int, machine_rank: int, main_process_ip: str, main_process_port: int.

        :param slurm_config: SlurmConfig, the slurm configuration dataclass
        :param slurm_params_kwargs: extra slurm arguments for the slurm configuration, defaults to {}
        :param slurm_submit_kwargs: extra slurm arguments for `srun` or `sbatch`, defaults to {}
        :param slurm_task_kwargs: extra arguments for the setting of distributed task, defaults to {}
        :param system_argv: the system arguments for the second launch in the distributed task (by default it will use the current system arguments `sys.argv[1:]`), defaults to None
        :return: the wrapped submit function with configured slurm paramters
        """
        slurm_fn = SlurmFunction(
            submit_fn=submit_fn,
        ).instantiate(
            slurm_config,
            slurm_params_kwargs,
            slurm_submit_kwargs,
            slurm_task_kwargs,
            system_argv,
        )
        return slurm_fn

    return wrapper


def slurm_fn(
    submit_fn: Callable,
):
    """A decorator to annoate a function to be run in slurm. The function decorated by this decorator should be launched in the way below.
    ```
    @slurm
    def run_in_slurm(*args, **kwargs):
        pass

    job = run_in_slurm[slurm_config](*args, **kwargs)
    ```
    The decorated function `submit_fn` is non-blocking now. To block and get the return value, you can call `job.result()`.
    """
    slurm_fn = SlurmFunction(submit_fn=submit_fn)

    return slurm_fn
