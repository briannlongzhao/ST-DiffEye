import os
import re
from pathlib import Path

import torch

from src.utils.train_utils import get_device

# Parameters that exist in checkpoints trained with the original research code
# but are never used by the released model.
LEGACY_KEY_PATTERNS = [
    re.compile(r"^scanpath_embedding\."),
    re.compile(r"^scanpath_head\."),
    re.compile(r"^layer_norm\."),
    re.compile(r"^blocks\.\d+\.adaLN_modulation\."),
]


def strip_legacy_keys(state_dict):
    return {k: v for k, v in state_dict.items() if not any(p.match(k) for p in LEGACY_KEY_PATTERNS)}


class ModelCheckpoint:
    def __init__(self, dirpath, every_n_epochs=50, save_last=True) -> None:
        self.dirpath = dirpath
        self.every_n_epochs = every_n_epochs
        self.save_last = save_last

        self.global_step = 0
        self.cur_step = 0
        self.global_epoch = 0

        Path(self.dirpath).mkdir(parents=True, exist_ok=True)

    def update_global_step(self):
        self.global_step += 1

    def set_cur_step(self, step):
        self.cur_step = step

    def update_global_epoch(self):
        self.global_epoch += 1

    def save_checkpoint(self, model, optimizer, save_last=False):
        ckpt_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "global_step": self.global_step,
            "global_epoch": self.global_epoch,
        }
        name = "last.pt" if save_last else f"epoch_{self.global_epoch}.pt"
        torch.save(ckpt_dict, Path(self.dirpath, name))

    def resume_checkpoint(self, model, optimizer, ckpt_path):
        assert os.path.exists(ckpt_path), f"{ckpt_path} does not exist, cwd: {os.getcwd()}"
        device = get_device()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(strip_legacy_keys(ckpt["model"]))
        model.to(device)

        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        self.global_step = ckpt.get("global_step", 0)
        self.global_epoch = ckpt.get("global_epoch", 0)

    def on_epoch_end(self, model, optimizer):
        if self.global_epoch % self.every_n_epochs == 0:
            self.save_checkpoint(model, optimizer)
        if self.save_last:
            self.save_checkpoint(model, optimizer, save_last=True)
        self.update_global_epoch()
