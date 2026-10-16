import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Sequence

import rich
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# Map a list to a string, e.g. tags -> run directory name.
OmegaConf.register_new_resolver("list_to_str", lambda lst: "_".join(lst) if lst else "", replace=True)


def extras(cfg: DictConfig) -> None:
    """Optionally silence warnings and print the resolved config tree."""
    if not cfg.get("extras"):
        return
    if cfg.extras.get("ignore_warnings"):
        warnings.filterwarnings("ignore")
    if cfg.extras.get("print_config"):
        print_config_tree(cfg, resolve=True, save_to_file=True)


def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = ("data", "model", "callbacks", "logger", "trainer", "paths", "extras"),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Print a DictConfig as a Rich tree, optionally saving it to the run output directory."""
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    queue = [field for field in print_order if field in cfg]
    queue += [field for field in cfg if field not in queue]
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)
        group = cfg[field]
        content = OmegaConf.to_yaml(group, resolve=resolve) if isinstance(group, DictConfig) else str(group)
        branch.add(rich.syntax.Syntax(content, "yaml"))
    rich.print(tree)
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


def log_hyperparameters(cfg: DictConfig, model, loggers) -> None:
    """Send the run config and parameter counts to the experiment loggers."""
    container = OmegaConf.to_container(cfg)
    hparams: Dict[str, Any] = {
        key: container.get(key)
        for key in ("data", "model", "diffusion", "trainer", "callbacks", "task_name", "tags", "ckpt_path", "seed")
    }
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for logger in loggers:
        logger.setup(hparams)
