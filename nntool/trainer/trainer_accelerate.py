import torch

from accelerate import Accelerator
from typing import Any, Dict, Tuple, Union
from dataclasses import dataclass

from .trainer_base import BaseTrainer


@dataclass
class AccelerateTrainer(BaseTrainer):
    accelerator: Accelerator

    def _trainer_init(self):
        # register inner trainer state
        self.accelerator.register_for_checkpointing(self.trainer_state)

        # prepare dataloaders
        self.dataloader, self.eval_dataloader, self.test_dataloader = (
            self._create_dataloaders(
                self.train_dataset, self.eval_dataset, self.test_dataset
            )
        )

        # prepare accelerator components
        (
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.dataloader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.dataloader,
        )

        if self.eval_dataloader is not None:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
        if self.test_dataloader is not None:
            self.test_dataloader = self.accelerator.prepare(self.test_dataloader)

        # print model architecture
        self.print(self.model)

        # load state if necessary
        if self.config.resume_trainer_from_dir is not None:
            self.load(self.config.resume_trainer_from_dir)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main_process(self):
        return self.accelerator.is_main_process

    @property
    def use_distributed(self):
        return self.accelerator.use_distributed

    def gather_for_metrics(self, *args, **kwargs):
        return self.accelerator.gather_for_metrics(*args, **kwargs)

    def print(self, *args, **kwargs):
        self.accelerator.print(*args, **kwargs)

    def save(self, milestone: Union[int, str]):
        # save checkpoint for all processes
        self.accelerator.save_state(
            f"{self.config.output_path}/checkpoints/checkpoints_{milestone}"
        )

        # save model checkpoints
        if self.accelerator.is_main_process:
            self.accelerator.save_model(
                self.model, f"{self.config.output_path}/models/model_{milestone}"
            )

        # wait for all processes to complete
        self.accelerator.wait_for_everyone()

    def load(
        self,
        state_dir: str,
    ):
        self.print("resume trainer from:", state_dir)

        # load the latest state from checkpoint folder
        self.accelerator.load_state(state_dir)

    def model_forward(self, batch_data):
        with self.accelerator.autocast():
            inputs, targets = batch_data
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
        return loss, outputs

    def _train_step(self, batch_data) -> Tuple[float, Dict[str, Any]]:
        with self.accelerator.accumulate(self.model):
            # train batch
            loss, _ = self.model_forward(batch_data)

            # train step
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        return loss.item(), {}
