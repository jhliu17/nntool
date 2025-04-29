import torch

from lightning.fabric import Fabric
from typing import Any, Dict, Tuple, Union
from dataclasses import dataclass

from .trainer_base import BaseTrainer


@dataclass
class FabricTrainer(BaseTrainer):
    """The FabricTrainer class is a subclass of the Trainer class, which uses the Lightning Fabric for training.

    Fabric doesn't provide a method for `gather_for_metrics`, one might need to implement a custom one.
    For example, if you want to gather predictions and references for metrics calculation, you can do it like this: https://github.com/huggingface/accelerate/blob/4b6be8991059f39a8df8893333d11c54bc51fc60/examples/by_feature/multi_process_metrics.py#L188

    samples_seen = 0
    for step, batch in enumerate(eval_dataloader):
        # We could avoid this line since we set the accelerator with `device_placement=True`.
        batch.to(accelerator.device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        # New Code #
        # First we check if it's a distributed system
        if accelerator.use_distributed:
            # Then see if we're on the last batch of our eval dataloader
            if step == len(eval_dataloader) - 1:
                # Last batch needs to be truncated on distributed systems as it contains additional samples
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
            else:
                # Otherwise we add the number of samples seen
                samples_seen += references.shape[0]
    """

    accelerator: Fabric

    def _trainer_init(self):
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
        ) = self.accelerator.setup(
            self.model,
            self.optimizer,
        )

        self.dataloader = self.accelerator.setup_dataloaders(self.dataloader)
        if self.eval_dataloader is not None:
            self.eval_dataloader = self.accelerator.setup_dataloaders(
                self.eval_dataloader
            )
        if self.test_dataloader is not None:
            self.test_dataloader = self.accelerator.setup_dataloaders(
                self.test_dataloader
            )

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
        return self.accelerator.is_global_zero

    @property
    def use_distributed(self):
        return self.accelerator.world_size > 1

    def gather_for_metrics(self, *args, **kwargs):
        raise Exception(
            "Fabric doesn't provide a method for `gather_for_metrics`. Please implement a custom one to truncate the repeated data."
        )

    def print(self, *args, **kwargs):
        self.accelerator.print(*args, **kwargs)

    def save(self, milestone: Union[int, str]):
        # save model checkpoints
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "trainer_state": self.trainer_state.state_dict(),
        }

        self.accelerator.save(
            f"{self.config.output_path}/checkpoints/checkpoints_{milestone}",
            state,
        )

    def load(
        self,
        state_dir: str,
    ):
        self.print("resume trainer from:", state_dir)

        # load the latest state from checkpoint folder
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
        }
        remainder = self.accelerator.load(state_dir, state)

        self.lr_scheduler.load_state_dict(remainder["lr_scheduler"])
        self.trainer_state.load_state_dict(remainder["trainer_state"])

    def model_forward(self, batch_data):
        with self.accelerator.autocast():
            inputs, targets = batch_data
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
        return loss, outputs

    def _train_step(self, batch_data) -> Tuple[float, Dict[str, Any]]:
        # Accumulate gradient at a time
        is_accumulating = (
            self.global_step % self.config.gradient_accumulation_steps != 0
        )

        with self.accelerator.no_backward_sync(self.model, enabled=is_accumulating):
            # train batch
            loss, _ = self.model_forward(batch_data)
            self.accelerator.backward(loss)

        # train step
        if not is_accumulating:
            self.accelerator.clip_gradients(
                self.model, self.optimizer, max_norm=self.config.max_grad_norm
            )
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        return loss.item(), {}
