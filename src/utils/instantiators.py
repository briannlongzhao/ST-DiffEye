import logging
from typing import List

import hydra
from omegaconf import DictConfig

from src.data.unified_datamodule import UnifiedDataModule

log = logging.getLogger(__name__)


def instantiate_datasets(datasets_cfg: DictConfig) -> List:
    """Instantiate every dataset in a ``data.train_datasets`` / ``data.test_datasets`` group."""
    datasets = []
    for name, ds_cfg in (datasets_cfg or {}).items():
        log.info(f"Loading dataset <{name}> ({ds_cfg.split} split)...")
        datasets.append(hydra.utils.instantiate(ds_cfg))
    return datasets


def build_datamodule(data_cfg: DictConfig, train: bool = False, test: bool = False) -> UnifiedDataModule:
    """Build the datamodule, loading only the requested splits."""
    return UnifiedDataModule(
        collator=hydra.utils.instantiate(data_cfg.collator),
        train_datasets=instantiate_datasets(data_cfg.train_datasets) if train else None,
        test_datasets=instantiate_datasets(data_cfg.test_datasets) if test else None,
        batch_size_train=data_cfg.batch_size_train,
        num_workers=data_cfg.num_workers,
    )


def instantiate_loggers(logger_cfg: DictConfig) -> List:
    """Instantiate the optional experiment loggers (e.g. Weights & Biases)."""
    loggers = []
    for _, lg_conf in (logger_cfg or {}).items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers
