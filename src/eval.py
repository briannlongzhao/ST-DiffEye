"""Generate gaze for a test set with a trained checkpoint and compute the evaluation metrics.

Example:
    python src/eval.py experiment=cocosearch18_tp data_dir=/path/to/data ckpt_path=checkpoints/COCOSearch18-TP.pt
"""
import logging

import hydra
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.callbacks.model_checkpoint import strip_legacy_keys
from src.utils.create_diffusion import create_diffusion
from src.utils.instantiators import build_datamodule
from src.utils.train_utils import get_device, seed_everything
from src.utils.utils import extras

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    extras(cfg)
    seed_everything(cfg.seed)

    datamodule = build_datamodule(cfg.data, test=True)
    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
    model.load_state_dict(strip_legacy_keys(ckpt["model"]))
    model.to(get_device())
    epoch = ckpt.get("global_epoch", 0)
    log.info(f"Loaded {cfg.ckpt_path} (epoch {epoch})")

    diffusion = create_diffusion(cfg.diffusion)
    evaluator = hydra.utils.instantiate(cfg.evaluation, datamodule=datamodule)
    evaluator.test(model, diffusion, epoch)
    log.info(f"Outputs saved to {cfg.evaluation.eval_root_path}")


if __name__ == "__main__":
    main()
