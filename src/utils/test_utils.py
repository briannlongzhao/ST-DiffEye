import torch
from tqdm import tqdm

from src.data.components.collator import PAD_VALUE

# Metrics are computed in a fixed 512x384 (W x H) frame.
METRIC_FRAME_H, METRIC_FRAME_W = 384, 512


def process_data_for_metrics_computation(fixations):
    """Convert normalized, padded scanpath tensors into per-key lists of {'X', 'Y', 'T'} (pixels, ms)."""
    scanpaths = {}
    for key in tqdm(fixations.keys()):
        entry = fixations[key]
        scanpaths[key] = {
            "size": (METRIC_FRAME_H, METRIC_FRAME_W),
            "scanpaths": [],
            "question_id": entry["question_id"],
            "dataset": entry["dataset"],
        }
        for scanpath in entry["scanpaths"]:
            if isinstance(scanpath, torch.Tensor):
                scanpath = scanpath.cpu().numpy()
            length = int((scanpath[:, 0] != PAD_VALUE).sum())
            x = list(scanpath[:length, 0] * METRIC_FRAME_W)
            y = list(scanpath[:length, 1] * METRIC_FRAME_H)
            t = list(scanpath[:length, 2] * 1000)  # s -> ms
            scanpaths[key]["scanpaths"].append({"X": x, "Y": y, "T": t})
    return scanpaths
