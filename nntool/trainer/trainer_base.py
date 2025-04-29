import torch
import wandb
import warnings

from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from tqdm import tqdm

from .trainer_utils import cycle_dataloader, divisible_by


@dataclass
class TrainerConfig:
    # training config
    train_batch_size: int = 512
    train_steps: int = 1_000
    train_num_workers: int = 2
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    checkpoint_interval: Optional[int] = None

    # evaluation config
    eval_interval: int = 1000
    eval_batch_size: int = 512
    eval_num_workers: int = 2
    eval_steps: int = -1

    # others
    random_seed: int = 42
    output_path: str = "outputs"
    resume_trainer_from_dir: Optional[str] = None

    def __post_init__(self):
        if self.checkpoint_interval is None:
            self.checkpoint_interval = self.eval_interval
            warnings.warn(
                "`checkpoint_interval` is not set. Using `eval_interval` as the default value for `checkpoint_interval`.",
                UserWarning,
            )

        if self.eval_interval % self.checkpoint_interval != 0:
            raise ValueError(
                f"`eval_interval` {self.eval_interval} should be divisible by `checkpoint_interval` {self.checkpoint_interval}"
            )


@dataclass
class TrainerState:
    global_step: int = 0

    def state_dict(self):
        state_dict = {
            "global_step": self.global_step,
        }
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.global_step = state_dict["global_step"]


@dataclass
class BaseTrainer(ABC):
    config: TrainerConfig
    accelerator: Any
    model: nn.Module
    loss_fn: nn.Module
    optimizer: optim.Optimizer
    lr_scheduler: optim.lr_scheduler.LRScheduler
    train_dataset: Dataset
    eval_dataset: Union[Dataset, None] = None
    test_dataset: Union[Dataset, None] = None
    log_to_wandb: bool = True
    trainer_state: TrainerState = field(default_factory=TrainerState)

    def __post_init__(self):
        self._config_check()
        self._trainer_init()

    def _config_check(self):
        pass

    def _trainer_init(self):
        # prepare dataloaders
        self.dataloader, self.eval_dataloader, self.test_dataloader = (
            self._create_dataloaders(
                self.train_dataset, self.eval_dataset, self.test_dataset
            )
        )

        # print model architecture
        self.print(self.model)

        # load state if necessary
        if self.config.resume_trainer_from_dir is not None:
            self.load(self.config.resume_trainer_from_dir)

    def _create_dataloaders(
        self,
        train_dataset: Dataset,
        eval_dataset: Union[Dataset, None],
        test_dataset: Union[Dataset, None],
    ) -> Tuple[DataLoader, Union[DataLoader, None], Union[DataLoader, None]]:
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.train_batch_size,
            num_workers=self.config.train_num_workers,
            shuffle=True,
        )
        eval_dataloader = (
            torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=self.config.eval_batch_size,
                num_workers=self.config.eval_num_workers,
                shuffle=False,
            )
            if eval_dataset is not None
            else None
        )
        test_dataloader = (
            torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.config.eval_batch_size,
                num_workers=self.config.eval_num_workers,
                shuffle=False,
            )
            if test_dataset is not None
            else None
        )
        return dataloader, eval_dataloader, test_dataloader

    @property
    def global_step(self) -> int:
        return self.trainer_state.global_step

    @global_step.setter
    def global_step(self, value: int):
        self.trainer_state.global_step = value

    @property
    def milestone(self) -> int:
        return self.global_step // self.config.checkpoint_interval

    def log(self, log_dict: dict, step: int = None, section: str = "train"):
        if self.is_main_process:
            if not self.log_to_wandb:
                return
            if step is None:
                step = self.global_step
            wandb.log({f"{section}/{k}": v for k, v in log_dict.items()}, step=step)

    @torch.no_grad()
    def eval_during_training(self, dataloader: torch.utils.data.DataLoader = None):
        """Evaluate the model during training. It will resume model state to training after evaluation."""
        results = self.eval(dataloader)
        self.model.train()
        return results

    @torch.no_grad()
    def eval(self, dataloader: torch.utils.data.DataLoader = None) -> Dict[str, float]:
        """Eval over the given dataloader.

        :param dataloader: the dataloader to be run, defaults to None (self.eval_dataloader). Please note that the
            dataloader should be prepared by the accelerator.
        :return: the evaluation metrics
        """
        self.model.eval()
        dl = self.eval_dataloader if dataloader is None else dataloader

        if dl is None:
            raise ValueError("No dataloader is provided for evaluation.")
        has_max_step = False if self.config.eval_steps == -1 else True
        results = self._eval_dataloader(dl, has_max_step, self.config.eval_steps)
        return results

    @torch.no_grad()
    def test(self, dataloader: torch.utils.data.DataLoader = None) -> Dict[str, float]:
        """Test over the given dataloader.

        :param dataloader: the dataloader to be run, defaults to None (self.test_dataloader). Please note that the
            dataloader should be prepared by the accelerator.
        :return: the evaluation metrics
        """
        dl = self.test_dataloader if dataloader is None else dataloader

        if dl is None:
            raise ValueError("No dataloader is provided for testing.")
        results = self.eval(dl)
        return results

    def train(self):
        self.model.train()

        dataloader = cycle_dataloader(self.dataloader)
        with tqdm(
            initial=self.global_step,
            total=self.config.train_steps,
            disable=not self.is_main_process,
        ) as self.pbar:
            while self.global_step < self.config.train_steps:
                # save checkpoint if necessary
                if self.global_step != 0 and divisible_by(
                    self.global_step, self.config.checkpoint_interval
                ):
                    self.print(
                        f"save checkpoint at step {self.global_step}, milestone {self.milestone}"
                    )
                    self.save(self.milestone)

                # do evaluation if necessary
                if self.global_step != 0 and divisible_by(
                    self.global_step, self.config.eval_interval
                ):
                    self.print(f"do evaluation at step {self.global_step}")
                    results = self.eval_during_training(self.eval_dataloader)
                    self.log(results, section="eval")
                    self.print("eval:", results)

                batch_data = next(dataloader)
                loss, additional_info = self._train_step(batch_data)

                # update state
                self.pbar.set_description(f"loss: {loss:.4f}")
                self.log(
                    {"loss": loss, "lr": self.lr_scheduler.get_last_lr()[0]},
                    section="train",
                )
                self.log(
                    additional_info,
                    section="train",
                )

                # update progress bar
                self.global_step += 1
                self.pbar.update(1)

        self.save("final")
        results = self.eval(self.test_dataloader)
        self.log(results, section="test")
        self.print("test:", results)
        self.print("final step:", self.global_step)
        self.print("training complete")

    @property
    @abstractmethod
    def device(self) -> torch.Device:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_main_process(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def use_distributed(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def gather_for_metrics(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def print(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, milestone: Union[int, str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(
        self,
        state_dir: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def model_forward(self, batch_data) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _eval_dataloader(
        self, dataloader: torch.utils.data.DataLoader, has_max_step: bool, max_step: int
    ) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def _train_step(self, batch_data) -> Tuple[float, Dict[str, Any]]:
        """Train the model over the given batch data.

        :param batch_data: batch data return by the dataloader
        :raises NotImplementedError: this function should be implemented by the subclass
        :return: the first item is the loss, the second item is the additional info
        """

        raise NotImplementedError
