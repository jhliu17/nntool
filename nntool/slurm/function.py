import os
import sys
import shlex
import submitit

from copy import deepcopy
from functools import partial
from submitit import Job, SlurmExecutor
from typing import Any, Callable, Literal, Tuple, Union, Dict, List, Optional

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
        submit_fn: Callable[..., Any],
        default_submit_fn_args: Optional[Tuple[Any]] = None,
        default_submit_fn_kwargs: Optional[Dict[str, Any]] = None,
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

        # slurm funcion is configured after calling `configure`
        self.__configured: bool = False
        self.__executor = None  # to be set up by `get_executor`

        # annotations here, will be set up after instantiation
        self.slurm_config: SlurmConfig
        self.slurm_params_kwargs: Dict[str, str]
        self.slurm_submit_kwargs: Dict[str, str]
        self.slurm_task_kwargs: Dict[str, str]
        self.system_argv: Optional[List[str]]
        self.pack_code_include_fn: Callable[[str, str], bool]
        self.pack_code_exclude_fn: Callable[[str, str], bool]

    def is_configured(self):
        """Whether the slurm function has been configured.

        :return: True if the slurm function has been configured, False otherwise
        """
        return self.submit_fn is not None and self.__configured

    def is_distributed(self):
        """Whether the slurm function is distributed.

        :return: True if the slurm function is distributed, False otherwise
        """
        return self.slurm_config.use_distributed_env

    def prepare_executor(self):
        slurm_config = self.slurm_config
        slurm_parameters_kwargs = self.slurm_params_kwargs
        slurm_submission_kwargs = self.slurm_submit_kwargs

        # select the cluster type, which is based on the submitit library
        # here we add a special mode called `exec` for running the job in the local machine
        # which is equivalent to the `debug` mode in the submitit library
        cluster_dispatch = {
            "slurm": None,
            "exec": "debug",
            "debug": "debug",
            "local": "local",
        }
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
        return executor

    def get_executor(
        self,
    ):
        if self.__executor is not None:
            return self.__executor
        else:
            executor = self.prepare_executor()
            self.__executor = executor
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

    def __mark_slurm_has_been_set_up(self):
        os.environ["NNTOOL_SLURM_HAS_BEEN_SET_UP"] = "1"

    def __update_slurm_kwargs(
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

    def configure(
        self,
        slurm_config: SlurmConfig,
        slurm_params_kwargs: Optional[Dict[str, str]] = None,
        slurm_submit_kwargs: Optional[Dict[str, str]] = None,
        slurm_task_kwargs: Optional[Dict[str, str]] = None,
        system_argv: Optional[List[str]] = None,
        pack_code_include_fn: Optional[Callable[[str, str], bool]] = None,
        pack_code_exclude_fn: Optional[Callable[[str, str], bool]] = None,
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

        slurm_fn.__update_slurm_kwargs(
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
        slurm_fn.__configured = True
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
            return self.configure(**slurm_configs)
        elif isinstance(slurm_configs, (list, tuple)):
            return self.configure(*slurm_configs)
        else:
            # will try to pass the slurm_configs as the first argument
            return self.configure(slurm_configs)

    def __before_submit(self, *args, **kwargs):
        """The hook function before submitting the job. It will pack the code and scripts to the slurm output folder if the `pack_code` is set to True in the slurm configuration. Only work before the first submit.

        :raises Exception: if the slurm function is not integrated
        """
        if self.slurm_has_been_set_up():
            return

        if not self.is_configured():
            raise Exception("A `SlurmFunction` should be configured before calling it.")

        # cache output path and current time into the environment variables for the second launch
        # this `hasattr` check is for backward compatiablity that the previous `SlurmConfig` does not have the
        # `output_path` and `output_path_date` attributes
        if hasattr(self.slurm_config, "output_path") and hasattr(
            self.slurm_config, "output_path_date"
        ):
            os.environ["NNTOOL_OUTPUT_PATH"] = self.slurm_config.output_path
            os.environ["NNTOOL_OUTPUT_PATH_DATE"] = self.slurm_config.output_path_date

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

    def __after_submit(
        self,
        submit_results: Union[Job, List[Job], Any] = None,
        *args,
        **kwargs,
    ):
        # get result to run program other than slurm mode
        if isinstance(submit_results, Job):
            if self.slurm_config.mode != "slurm":
                submit_results.results()
        elif (
            isinstance(submit_results, list)
            and submit_results
            and isinstance(submit_results[0], Job)
        ):
            if self.slurm_config.mode != "slurm":
                for job in submit_results:
                    job.results()
        else:
            pass

    def __call__(self, *submit_fn_args, **submit_fn_kwargs) -> Union[Job, Any]:
        """Run the submit_fn with the given arguments and keyword arguments. The function is non-blocking in the mode of `slurm`, while other modes cause blocking. If there is no given arguments or keyword arguments, the default arguments and keyword arguments will be used.

        :raises Exception: if the submit_fn is not set up
        :return: Slurm Job or the return value of the submit_fn
        """
        self.__before_submit()
        submit_strategy = self.__dispatch_submit_strategy("submit")
        submit_results = submit_strategy(*submit_fn_args, **submit_fn_kwargs)
        self.__after_submit(submit_results)
        return submit_results

    def __dispatch_submit_strategy(
        self,
        submit_mode: Literal["submit", "map_array"] = "submit",
        *submit_fn_args,
        **submit_fn_kwargs,
    ) -> Callable[..., Union[Job, List[Job], Any]]:
        if submit_mode == "submit":
            if self.is_distributed():
                return self.__distributed_submit
            else:
                return self.__submit
        elif submit_mode == "map_array":
            if self.is_distributed():
                raise Exception("Distributed job does not support `map_array` mode.")
            else:
                return self.__submit_map_array
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
        self.__before_submit()
        submit_strategy = self.__dispatch_submit_strategy("map_array")
        submit_results = submit_strategy(*submit_fn_args, **submit_fn_kwargs)
        self.__after_submit(submit_results)
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

    def __get_submit_args(
        self,
        *submit_fn_args,
        **submit_fn_kwargs,
    ):
        submit_fn_args = (
            self.default_submit_fn_args if not submit_fn_args else submit_fn_args
        )
        submit_fn_kwargs = (
            self.default_submit_fn_kwargs if not submit_fn_kwargs else submit_fn_kwargs
        )
        return submit_fn_args, submit_fn_kwargs

    def __submit(
        self,
        *submit_fn_args,
        **submit_fn_kwargs,
    ) -> Job:
        submit_fn_args, submit_fn_kwargs = self.__get_submit_args(
            *submit_fn_args, **submit_fn_kwargs
        )
        executor = self.get_executor()
        self.__mark_slurm_has_been_set_up()
        job = executor.submit(self.submit_fn, *submit_fn_args, **submit_fn_kwargs)
        return job

    def __submit_map_array(
        self,
        *submit_fn_args,
        **submit_fn_kwargs,
    ) -> List[Job]:
        submit_fn_args, submit_fn_kwargs = self.__get_submit_args(
            *submit_fn_args, **submit_fn_kwargs
        )
        executor = self.get_executor()
        self.__mark_slurm_has_been_set_up()
        job = executor.map_array(self.submit_fn, *submit_fn_args, **submit_fn_kwargs)
        return job

    def __distributed_submit(
        self,
        *submit_fn_args,
        **submit_fn_kwargs,
    ) -> Union[Job, Any]:
        submit_fn_args, submit_fn_kwargs = self.__get_submit_args(
            *submit_fn_args, **submit_fn_kwargs
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
            # but directly use `sbatch`` to submit the second job without any issues
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
            self.__mark_slurm_has_been_set_up()
            job = executor.submit(task)
            return job
        else:
            return self.submit_fn(*submit_fn_args, **submit_fn_kwargs)
