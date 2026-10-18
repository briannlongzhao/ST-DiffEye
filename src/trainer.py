import logging

import torch
from tqdm import tqdm

from src.utils.train_utils import get_device

log = logging.getLogger(__name__)


class Trainer:
    """Minimal single-GPU training loop for the gaze diffusion model."""

    def __init__(self, max_epochs, learning_rate=1e-4, weight_decay=1e-2, schedule_sampler=None, model_checkpoint=None, loggers=None):
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.schedule_sampler = schedule_sampler
        self.model_checkpoint = model_checkpoint
        self.loggers = loggers or []
        self.device = get_device()

    def train(self, model, diffusion, datamodule, ckpt_path=None):
        self.opt = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        start_epoch = 0
        if ckpt_path is not None:
            self.model_checkpoint.resume_checkpoint(model, self.opt, ckpt_path)
            start_epoch = self.model_checkpoint.global_epoch
            log.info(f"Resumed from {ckpt_path} at epoch {start_epoch}")

        model.to(self.device)
        model.train()
        for epoch in range(start_epoch, self.max_epochs):
            log.info(f"Epoch {epoch}/{self.max_epochs - 1}")
            self._train_one_epoch(model, diffusion, datamodule)
            self.log({"epoch": self.model_checkpoint.global_epoch})
            self.model_checkpoint.on_epoch_end(model, self.opt)

    def _train_one_epoch(self, model, diffusion, datamodule):
        for step, batch in enumerate(tqdm(datamodule.train_dataloader())):
            x = batch["trajectory"].to(self.device)
            t, weights = self.schedule_sampler.sample(x.shape[0], self.device)
            model_kwargs = dict(
                y=batch["img"].to(self.device),
                task_embedding=batch["task_embedding"].to(self.device),
                padding_mask=batch["trajectory_padding_mask"].to(self.device),
                fixation_indices=batch["traj_fixation_index"].to(self.device),
            )
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            self.schedule_sampler.update_with_local_losses(t, loss_dict["loss"].detach())
            loss = (loss_dict["loss"] * weights).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            self.log({f"train/{k}": v.mean().item() for k, v in loss_dict.items()})
            self.model_checkpoint.set_cur_step(step)
            self.model_checkpoint.update_global_step()

    def log(self, log_dict):
        for logger in self.loggers:
            logger.log(log_dict)
