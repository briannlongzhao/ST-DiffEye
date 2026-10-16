"""Train ST-DiffEye.

Example:
    python src/train.py experiment=cocosearch18_tp data_dir=/path/to/data
"""
import logging

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model.components.step_sample import LossSecondMomentResampler
from src.utils.create_diffusion import create_diffusion
from src.utils.instantiators import build_datamodule, instantiate_loggers
from src.utils.train_utils import seed_everything
from src.utils.utils import extras, log_hyperparameters

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    extras(cfg)
    if cfg.get("seed") is not None:
        seed_everything(cfg.seed)

    datamodule = build_datamodule(cfg.data, train=True)
    model = hydra.utils.instantiate(cfg.model)
    diffusion = create_diffusion(cfg.diffusion)
    loggers = instantiate_loggers(cfg.get("logger"))
    if loggers:
        log_hyperparameters(cfg, model, loggers)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        model_checkpoint=hydra.utils.instantiate(cfg.checkpoint),
        schedule_sampler=LossSecondMomentResampler(diffusion),
        loggers=loggers,
    )
    trainer.train(model, diffusion, datamodule, ckpt_path=cfg.get("ckpt_path"))
    log.info(f"Checkpoints saved to {cfg.checkpoint.dirpath}")


if __name__ == "__main__":
    main()
