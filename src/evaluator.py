import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from src.data.components.collator import PAD_VALUE
from src.data.unified_datamodule import UnifiedDataModule
from src.gazetools.display import save_image_scanpaths
from src.gazetools.metrics.run_metrics import run_metrics
from src.utils.check_fixations import check_fixations
from src.utils.test_utils import process_data_for_metrics_computation
from src.utils.train_utils import get_device, seed_everything

log = logging.getLogger(__name__)

SEARCH_DATASETS = ("cocosearch18_edf_tp", "cocosearch18_edf_ta")

# Sequence-score cluster file, located in each dataset's root folder.
CLUSTER_FILES = {
    "cocosearch18_edf_tp": "clusters_cocosearch18_tp_512_384.npy",
    "cocosearch18_edf_ta": "clusters_cocosearch18_ta_512_384.npy",
    "cocofreeview_edf": "clusters_coco_freeview_512_384.npy",
    "mit1003_edf": "clusters_mit1003_512_384.npy",
}

# Output keys produced by run_metrics for each requested metric, used to order the json.
METRIC_OUTPUT_KEYS = {
    "levenshtein_distance": ["LevenshteinDistance", "LevenshteinDistanceBest"],
    "discrete_frechet_distance": ["DiscreteFrechetDistance", "DiscreteFrechetDistanceBest"],
    "dynamic_time_warping": ["DynamicTimeWarping", "DynamicTimeWarpingBest"],
    "time_delay_embedding": ["TimeDelayEmbedding", "TimeDelayEmbeddingBest"],
    "multi_match": [
        "MM_Vector", "MM_VectorBest", "MM_Direction", "MM_DirectionBest", "MM_Length", "MM_LengthBest",
        "MM_Position", "MM_PositionBest", "MM_Duration", "MM_DurationBest", "MultiMatch", "MultiMatchBest",
    ],
    "scan_match": ["ScanMatch", "ScanMatchTime"],
    "sequence_score": ["SequenceScore", "SequenceScoreTime"],
    "semantic_sequence_score": ["SemanticSequenceScore", "SemanticSequenceScoreTime"],
}


def convert_trajectory_to_scanpath(trajectory_batch, max_len=16, padding_value=PAD_VALUE, device="cuda", normalization_size=None):
    """Convert generated trajectories into scanpaths with the fixation detection algorithm.

    Args:
        trajectory_batch: (batch, seq_len, 3) tensor of normalized (x, y, dt); padded rows hold ``padding_value``.
        max_len: scanpaths are padded / truncated to this many fixations.
        normalization_size: (width, height) of the frame used to normalize coordinates.

    Returns:
        (batch, max_len, 3) tensor of normalized fixation (x, y, duration in seconds).
    """
    img_width, img_height = normalization_size
    converted = []
    for traj in trajectory_batch:
        traj_xy = traj[:, :2].cpu().numpy().copy()
        valid = traj_xy[:, 0] != padding_value
        if not valid.any():
            converted.append(torch.full((1, 3), padding_value))
            continue
        traj_xy = traj_xy[valid]
        traj_xy[:, 0] *= img_width
        traj_xy[:, 1] *= img_height
        _, fix_data, _ = check_fixations(traj_xy)
        if len(fix_data.medianXY) > 0:
            scanpath = np.column_stack([
                fix_data.medianXY[:, 0] / img_width,
                fix_data.medianXY[:, 1] / img_height,
                fix_data.duration / 1000.0,  # ms -> s
            ])
            converted.append(torch.from_numpy(scanpath).float())
        else:
            converted.append(torch.full((1, 3), padding_value))

    padded = []
    for s in converted:
        if s.shape[0] < max_len:
            s = torch.cat([s, torch.full((max_len - s.shape[0], 3), padding_value)], dim=0)
        padded.append(s[:max_len])
    return torch.stack(padded).to(device)


class Evaluator:
    """Generates gaze for the test split, converts it to scanpaths and computes metrics.

    ``data_to_extract`` selects the outputs:
      - ``preds``: generate (or reload) trajectories and scanpaths -> ``generations_epoch_<E>/``
      - ``metrics``: compute CRPS / Mean / Best / KLD metrics -> ``metrics_epoch_<E>/``
      - ``qualitatives``: draw generated scanpaths on the images -> ``qualitatives/``
    """

    def __init__(
        self,
        eval_root_path: str,
        metrics_to_compute: List[str],
        data_to_extract: List[str],
        datamodule: UnifiedDataModule,
        seed: Optional[int] = None,
        scanpath_eval_max_len: int = 16,
        token_validity_threshold: float = 0.5,
    ) -> None:
        self.eval_root_path = eval_root_path
        self.datamodule = datamodule
        self.metrics_to_compute = list(metrics_to_compute)
        self.data_to_extract = list(data_to_extract)
        self.seed = seed
        self.device = get_device()
        self.scanpath_eval_max_len = scanpath_eval_max_len
        self.token_validity_threshold = float(token_validity_threshold)

    @property
    def dataset(self):
        return self.datamodule.test_datasets[0]

    def normalization_size(self, original_img_size):
        """Frame used to normalize x/y: a fixed screen frame if the dataset defines one, else the image size."""
        if hasattr(self.dataset, "frame_width") and hasattr(self.dataset, "frame_height"):
            return float(self.dataset.frame_width), float(self.dataset.frame_height)
        return float(original_img_size[0]), float(original_img_size[1])

    @property
    def dataset_tag(self):
        return self.dataset.__class__.__name__ + "_test"

    def _generation_paths(self, epoch):
        root = Path(self.eval_root_path, f"generations_epoch_{epoch}")
        return root / f"generations_{self.dataset_tag}.pkl", root / f"original_{self.dataset_tag}.pkl"

    def test(self, model, diffusion, epoch):
        preds, target = None, None
        if "preds" in self.data_to_extract:
            preds_pkl, target_pkl = self._generation_paths(epoch)
            if preds_pkl.exists() and target_pkl.exists():
                with open(preds_pkl, "rb") as f:
                    preds = pickle.load(f)
                with open(target_pkl, "rb") as f:
                    target = pickle.load(f)
                log.info(f"Loaded existing generations from {preds_pkl.parent}")
            else:
                preds, target = self.extract_predictions(model, diffusion, epoch)
        if "qualitatives" in self.data_to_extract:
            self.save_qualitatives(epoch, preds)
        if "metrics" in self.data_to_extract:
            self.compute_metrics(epoch, preds, target)

    def _apply_predicted_padding(self, pred_xyz, token_logits):
        """Pad every token after the first one predicted as invalid."""
        valid = (nn.Softmax(dim=-1)(token_logits)[..., 1] >= self.token_validity_threshold).long()
        seq_lengths = torch.cumprod(valid, dim=-1).sum(-1)
        mask = torch.arange(pred_xyz.shape[1], device=pred_xyz.device).unsqueeze(0) >= seq_lengths.unsqueeze(1)
        padded = pred_xyz.clone()
        padded[mask.unsqueeze(2).repeat(1, 1, pred_xyz.shape[2])] = PAD_VALUE
        return padded

    def extract_predictions(self, model, diffusion, epoch):
        log.info(f"Generating test-set predictions (epoch {epoch})...")
        model.to(self.device)
        model.eval()
        if self.seed is not None:
            # Re-seed so generations do not depend on how many random numbers were used before.
            seed_everything(self.seed)

        target, preds = {}, {}
        data_loader = self.datamodule.test_dataloader()
        with torch.no_grad():
            for batch in tqdm(data_loader, total=len(data_loader)):
                img_filename = batch["img_filename"][0]
                original_img_size = batch["original_img_size"][0]
                gt_scanpaths = batch["scanpath"].to(self.device)
                dataset_name = batch["dataset"][0]
                key = (img_filename, batch["task"][0]) if dataset_name in SEARCH_DATASETS else img_filename

                if key not in target:
                    target[key] = {"scanpaths": gt_scanpaths}
                else:
                    target[key]["scanpaths"] = torch.cat((target[key]["scanpaths"], gt_scanpaths), dim=0)
                target[key].update(size=original_img_size, dataset=dataset_name, question_id="")
                if key in preds:
                    continue

                # One generated sample per human viewer of this stimulus.
                num_viewers = batch["num_viewers"][0]
                model_kwargs = dict(
                    y=batch["img"].repeat(num_viewers, 1, 1).to(self.device),
                    task_embedding=batch["task_embedding"].to(self.device).repeat(num_viewers, 1),
                )
                initial_noise = torch.randn(num_viewers, model.max_len, model.diffusion_latent_dim).to(self.device)
                samples = diffusion.p_sample_loop(
                    model,
                    initial_noise.shape,
                    initial_noise,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=True,
                    device=self.device,
                )

                pred_xyz = model.get_coords_and_time(samples)[:, :, :3]
                pred_trajectories = self._apply_predicted_padding(pred_xyz, model.token_validity_predictor(samples))
                pred_scanpaths = convert_trajectory_to_scanpath(
                    pred_trajectories,
                    max_len=self.scanpath_eval_max_len,
                    device=self.device,
                    normalization_size=self.normalization_size(original_img_size),
                )
                preds[key] = {
                    "size": original_img_size,
                    "scanpaths": pred_scanpaths,
                    "trajectories": pred_trajectories,
                    "question_id": "",
                    "dataset": dataset_name,
                }

        preds_pkl, target_pkl = self._generation_paths(epoch)
        os.makedirs(preds_pkl.parent, exist_ok=True)
        with open(preds_pkl, "wb") as f:
            pickle.dump(preds, f)
        with open(target_pkl, "wb") as f:
            pickle.dump(target, f)
        log.info(f"Saved generations to {preds_pkl.parent}")
        return preds, target

    def save_qualitatives(self, epoch, preds=None):
        if preds is None:
            preds_pkl, _ = self._generation_paths(epoch)
            if not preds_pkl.exists():
                log.error("No generations found; run with data_to_extract including 'preds' first.")
                return
            with open(preds_pkl, "rb") as f:
                preds = pickle.load(f)

        root_path = Path(self.eval_root_path, "qualitatives", self.dataset_tag)
        log.info(f"Saving qualitatives to {root_path}")
        img_root = Path(self.dataset.root_path, getattr(self.dataset, "img_dir", "images"))
        for key, data in preds.items():
            if isinstance(key, tuple):
                img_name, task = key
                dir_name = img_name.rsplit(".", 1)[0] + "_" + task
                img_path = img_root / task / img_name
            else:
                dir_name = key.rsplit(".", 1)[0]
                img_path = img_root / key
            os.makedirs(root_path / dir_name, exist_ok=True)
            pil_img = Image.open(img_path)
            width, height = pil_img.size
            for idx, s in enumerate(data["scanpaths"]):
                s = s.cpu().numpy()
                length = int((s[:, 0] != PAD_VALUE).sum())
                save_image_scanpaths(
                    pil_img,
                    x=s[:length, 0] * width,
                    y=s[:length, 1] * height,
                    t=s[:length, 2] * 1000,  # s -> ms
                    save_path=root_path / dir_name / f"subject_{idx}.jpg",
                )

    def compute_metrics(self, epoch, pred_scanpaths=None, gt_scanpaths=None):
        if pred_scanpaths is None or gt_scanpaths is None:
            preds_pkl, target_pkl = self._generation_paths(epoch)
            if not preds_pkl.exists():
                raise FileNotFoundError(f"No generations found at {preds_pkl}; include 'preds' in data_to_extract.")
            with open(preds_pkl, "rb") as f:
                pred_scanpaths = pickle.load(f)
            with open(target_pkl, "rb") as f:
                gt_scanpaths = pickle.load(f)

        log.info("Computing scanpath metrics...")
        gt_scanpaths = process_data_for_metrics_computation(gt_scanpaths)
        pred_scanpaths = process_data_for_metrics_computation(pred_scanpaths)
        data_root = self.dataset.root_path

        ss_clusters = None
        if "sequence_score" in self.metrics_to_compute:
            ss_clusters = np.load(Path(data_root, CLUSTER_FILES[self.dataset.name]), allow_pickle=True).item()

        metrics, *extra_outputs = run_metrics(
            gt_scanpaths, pred_scanpaths, self.datamodule, self.metrics_to_compute, ss_clusters, data_root=data_root
        )
        metrics_dict = self._order_metrics_dict(self._merge_best_metrics(metrics.to_dict()))

        out_dir = Path(self.eval_root_path, f"metrics_epoch_{epoch}")
        os.makedirs(out_dir, exist_ok=True)
        out_file = out_dir / f"metrics_{self.dataset_tag}.json"
        with open(out_file, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        log.info(f"Saved metrics to {out_file}")
        return metrics_dict

    @staticmethod
    def _merge_best_metrics(metrics_dict):
        """Combine '<metric>' and '<metric>Best' rows into one entry with Mean/Best/Human/KLD/CRPS."""
        merged = {}
        for name, values in metrics_dict.items():
            if name.endswith("Best"):
                continue
            best_name = f"{name}Best"
            if best_name not in metrics_dict:
                merged[name] = values
                continue
            entry = {"ModelMean": values.get("Model"), "ModelBest": metrics_dict[best_name].get("Model")}
            for k in ("Human", "KLD", "CRPS"):
                if k in values:
                    entry[k] = values[k]
            merged[name] = entry
        return merged

    def _order_metrics_dict(self, metrics_dict):
        ordered = []
        for metric in self.metrics_to_compute:
            ordered += [k for k in METRIC_OUTPUT_KEYS.get(metric, []) if k in metrics_dict and k not in ordered]
        ordered += [k for k in metrics_dict if k not in ordered]
        return {k: metrics_dict[k] for k in ordered}
