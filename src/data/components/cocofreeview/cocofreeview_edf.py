import json
import pickle
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.utils.check_fixations import check_fixations
from src.utils.load_edf import read_edfasc


class COCOFreeviewEDFDataset:
    def __init__(
        self,
        name: str,
        root_path: str,
        split: str,
        task_embeddings_path: str,
        threshold: float = -1.0,
        threshold_no_dur: float = -1.0,
        use_cache: bool = True,
        sample_rate: int = 1000,
        img_features_dir: str = "dinov2_base_timm_image_features",
        max_samples: int = -1,
        head_align_distance_threshold_px: float = 10.0,
    ) -> None:
        self.name = name
        self.root_path = root_path
        self.split = split
        # Sequence-score similarity thresholds used by the evaluation metrics.
        self.threshold = threshold
        self.threshold_no_dur = threshold_no_dur
        self.img_features_dir = img_features_dir
        self.sample_rate = sample_rate
        self.head_align_distance_threshold_px = head_align_distance_threshold_px
        # Gaze coordinates are recorded in the 1680x1050 screen frame.
        self.frame_width = 1680.0
        self.frame_height = 1050.0
        self.cache_path = Path(root_path) / "cocofreeview_edf_cache.pkl"

        json_split = "validation" if split == "valid" else split
        self.canonical_split = "validation" if json_split == "validation" else json_split
        with open(Path(self.root_path, f"COCOFreeView_fixations_{json_split}.json"), "rb") as f:
            loaded = json.load(f)

        # Freeview JSONs may still carry split tags. Keep only requested split when present.
        samples = [s for s in loaded if s.get("split", self.canonical_split) == self.canonical_split]
        if max_samples > 0:
            samples = samples[:max_samples]

        # Deterministic ordering and duplicate check.
        samples = sorted(samples, key=lambda x: (x["name"], self._normalize_subject(x.get("subject", -1))))
        sample_keys = [(s["name"], self._normalize_subject(s.get("subject", -1))) for s in samples]
        duplicate_keys = [(k, c) for k, c in Counter(sample_keys).items() if c > 1]
        if duplicate_keys:
            duplicate_preview = ", ".join([f"{k} x{c}" for k, c in duplicate_keys[:10]])
            raise ValueError(
                "Found duplicate fixation keys for (name, subject). "
                f"Count: {len(duplicate_keys)}. Examples: {duplicate_preview}"
            )
        self.samples = samples

        # Load EDF trajectories and index by (image, subject) key variants.
        traj_data = self._load_or_build_traj_cache(use_cache=use_cache)
        traj_index = self._index_trajectories_by_img_subject(traj_data)

        # Join fixation + EDF trajectory.
        joined_samples = []
        dropped_missing_keys = []
        kept_with_dummy_keys = []
        for sample_idx, sample in enumerate(tqdm(self.samples, desc="Processing COCOFreeview EDF samples")):
            match_key = self._match_sample_to_traj_key(sample, traj_index)
            if match_key is None:
                missing_key = (sample["name"], sample.get("subject", None))
                # Missing trajectory policy:
                # - train/validation: drop sample
                # - test: keep sample with dummy trajectory fields
                if self.canonical_split == "test":
                    processed = self._attach_dummy_trajectory(sample.copy())
                    joined_samples.append(processed)
                    kept_with_dummy_keys.append(missing_key)
                else:
                    dropped_missing_keys.append(missing_key)
                continue

            trajectory = traj_index[match_key]
            sample = sample.copy()
            processed = self._attach_processed_trajectory(sample, trajectory)
            joined_samples.append(processed)

        if dropped_missing_keys:
            unique_missing = sorted(set(dropped_missing_keys))
            preview = ", ".join([str(k) for k in unique_missing[:10]])
            print(
                "COCOFreeviewEDF: dropping samples without matched EDF trajectory. "
                f"Dropped: {len(dropped_missing_keys)} / {len(self.samples)}. "
                f"Unique missing keys: {len(unique_missing)}. "
                f"split={self.canonical_split}. "
                f"Examples: {preview}"
            )
        if kept_with_dummy_keys:
            unique_dummy = sorted(set(kept_with_dummy_keys))
            preview = ", ".join([str(k) for k in unique_dummy[:10]])
            print(
                "COCOFreeviewEDF: keeping test samples without matched EDF trajectory "
                "using dummy trajectory fields. "
                f"Kept-with-dummy: {len(kept_with_dummy_keys)} / {len(self.samples)}. "
                f"Unique keys: {len(unique_dummy)}. "
                f"Examples: {preview}"
            )
        self.samples = joined_samples

        self.subject_per_img = {}
        for s in self.samples:
            img_name = s["name"]
            self.subject_per_img[img_name] = self.subject_per_img.get(img_name, 0) + 1

        embeddings = np.load(task_embeddings_path, allow_pickle=True).item()
        self.task_embedding = torch.from_numpy(embeddings[""])  # free viewing: empty task

    def __getitem__(self, index: int):
        sample = self.samples[index]
        img_filename = sample["name"]
        num_viewers = self.subject_per_img[img_filename]

        original_img_size = Image.open(Path(self.root_path, "images", Path(img_filename))).size
        img_features_path = Path(self.root_path, self.img_features_dir, Path(img_filename).stem + ".pth")
        img_feats = torch.load(img_features_path).unsqueeze(0)

        # Normalize coordinates by the screen frame and convert durations from ms to s.
        trajectory = np.stack((sample["X_traj"], sample["Y_traj"], sample["T_traj"]), axis=1).astype(np.float32)
        trajectory[:, 0] /= self.frame_width
        trajectory[:, 1] /= self.frame_height
        trajectory = torch.from_numpy(trajectory).float()
        trajectory[:, 2] /= 1000.0

        scanpath = np.stack(
            (np.asarray(sample["X"], dtype=np.float32), np.asarray(sample["Y"], dtype=np.float32), np.asarray(sample["T"], dtype=np.float32)),
            axis=1,
        ).astype(np.float32)
        scanpath[:, 0] /= self.frame_width
        scanpath[:, 1] /= self.frame_height
        scanpath = torch.from_numpy(scanpath).float()
        scanpath[:, 2] /= 1000.0

        return {
            "img_filename": img_filename,
            "original_img_size": original_img_size,
            "img": img_feats.squeeze(),
            "scanpath": scanpath,
            "task": "",
            "task_embedding": self.task_embedding,
            "num_viewers": num_viewers,
            "dataset": self.name,
            "question_id": "",
            "trajectory": trajectory,
            "traj_fixation_index": torch.from_numpy(np.asarray(sample["traj_fixation_index"], dtype=np.float32)),
        }

    def __len__(self):
        return len(self.samples)

    def _load_or_build_traj_cache(self, use_cache: bool):
        if use_cache and self.cache_path.exists():
            cached = None
            try:
                with open(self.cache_path, "rb") as f:
                    cached = pickle.load(f)
            except Exception:
                pass

            if cached is not None and len(cached) > 0:
                return cached

        all_data = []
        all_asc_files = list(Path(self.root_path).rglob("*.asc"))
        for file_path in tqdm(all_asc_files, desc="Loading COCOFreeview EDF ASC files"):
            try:
                # COCO-Freeview EDF ASC metadata does not include search-specific
                # fields (category/condition), so parse in freeview mode.
                data = read_edfasc(file_path, task="freeview")
                all_data.extend(data)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                traceback.print_exc()

        with open(self.cache_path, "wb") as f:
            pickle.dump(all_data, f)
        return all_data

    @staticmethod
    def _normalize_subject(subject):
        try:
            return int(subject)
        except Exception:
            return str(subject)

    @staticmethod
    def _image_key_variants(img_name):
        p = Path(str(img_name))
        stem = p.stem.lower()
        base = p.name.lower()
        variants = {base, stem, f"{stem}.jpg", f"{stem}.jpeg", f"{stem}.png"}
        return variants

    def _index_trajectories_by_img_subject(self, trajectories):
        indexed = {}
        duplicate = []
        for d in trajectories:
            subject = self._normalize_subject(d.get("subject", -1))
            for img_variant in self._image_key_variants(d.get("img_name", "")):
                key = (img_variant, subject)
                if key in indexed:
                    duplicate.append(key)
                    continue
                indexed[key] = d
        if duplicate:
            uniq = sorted(set(duplicate))
            preview = ", ".join([str(k) for k in uniq[:10]])
            raise ValueError(
                "Found duplicate EDF trajectory keys for (image, subject). "
                f"Unique duplicates: {len(uniq)}. Examples: {preview}"
            )
        return indexed

    def _match_sample_to_traj_key(self, sample, traj_index):
        subject = self._normalize_subject(sample.get("subject", -1))
        for img_variant in self._image_key_variants(sample["name"]):
            key = (img_variant, subject)
            if key in traj_index:
                return key
        return None

    def _attach_processed_trajectory(self, sample, trajectory):
        sample_rate = self.sample_rate
        unit = 1 / sample_rate * 1000  # ms per raw sample at source frequency

        raw_t = np.asarray(trajectory["t"], dtype=np.float64)
        if len(raw_t) > 0:
            raw_t = np.array([unit] + (unit * (raw_t[1:] - raw_t[:-1])).tolist(), dtype=np.float64)

        raw_x = np.asarray(trajectory["x"], dtype=np.float64)
        raw_y = np.asarray(trajectory["y"], dtype=np.float64)
        original_width, original_height = Image.open(Path(self.root_path, "images", sample["name"])).size

        # Align raw EDF trajectory to freeview canonical frame used by annotations.
        raw_x = raw_x * (self.frame_width / original_width)
        raw_y = raw_y * (self.frame_height / original_height)
        raw_traj = self._as_trajectory(raw_x, raw_y, raw_t)
        raw_traj_uncut = raw_traj

        gt_scanpath_xy = np.column_stack(
            (np.asarray(sample["X"], dtype=np.float64), np.asarray(sample["Y"], dtype=np.float64))
        )
        gt_total_duration_ms = float(np.sum(np.asarray(sample["T"], dtype=np.float64)))

        # Trim the trajectory to the span covered by the human scanpath, then resample to 240 Hz.
        raw_traj, _, raw_traj_discarded_tail = self.cut_trajectory(
            traj=raw_traj,
            gt_scanpath_xy=gt_scanpath_xy,
            gt_total_duration_ms=gt_total_duration_ms,
            distance_threshold_px=self.head_align_distance_threshold_px,
        )

        traj = self.downsample_trajectory(raw_traj, input_freq=sample_rate, output_freq=240)
        discarded_tail = self.downsample_trajectory(raw_traj_discarded_tail, input_freq=sample_rate, output_freq=240)
        if len(traj) == 0:
            traj = self.downsample_trajectory(raw_traj_uncut, input_freq=sample_rate, output_freq=240)
            discarded_tail = np.empty((0, 3), dtype=np.float64)

        traj, discarded_tail = self._recover_tail_if_scanpath_shorter(
            kept_traj=traj,
            discarded_tail=discarded_tail,
            gt_scanpath_xy=gt_scanpath_xy,
            original_width=original_width,
            original_height=original_height,
            distance_threshold_px=self.head_align_distance_threshold_px,
        )

        x, y, t = self._split_trajectory(traj)
        sample["X_traj"] = x
        sample["Y_traj"] = y
        sample["T_traj"] = t
        sample["traj_fixation_index"] = self.build_traj_fixation_index(
            trajectory_xy=np.column_stack((x, y)),
            scanpath_data=sample,
        )
        return sample

    def _attach_dummy_trajectory(self, sample):
        """Attach minimal placeholder trajectory fields for unmatched test samples."""
        fix_x = np.asarray(sample.get("X", []), dtype=np.float64)
        fix_y = np.asarray(sample.get("Y", []), dtype=np.float64)
        fix_t = np.asarray(sample.get("T", []), dtype=np.float64)

        if len(fix_x) > 0 and len(fix_y) > 0:
            x0 = float(fix_x[0])
            y0 = float(fix_y[0])
            t0 = float(fix_t[0]) if len(fix_t) > 0 else 1.0
            if (not np.isfinite(t0)) or t0 <= 0:
                t0 = 1.0
        else:
            x0 = self.frame_width * 0.5
            y0 = self.frame_height * 0.5
            t0 = 1.0

        sample["X_traj"] = np.array([x0], dtype=np.float64)
        sample["Y_traj"] = np.array([y0], dtype=np.float64)
        sample["T_traj"] = np.array([t0], dtype=np.float64)
        sample["traj_fixation_index"] = self.build_traj_fixation_index(
            trajectory_xy=np.column_stack((sample["X_traj"], sample["Y_traj"])),
            scanpath_data=sample,
        )
        return sample

    def _as_trajectory(self, x, y, t):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        if len(x) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.column_stack((x, y, t))

    def _split_trajectory(self, traj):
        traj = np.asarray(traj, dtype=np.float64)
        if len(traj) == 0:
            empty = np.array([], dtype=np.float64)
            return empty, empty, empty
        return traj[:, 0], traj[:, 1], traj[:, 2]

    def downsample_trajectory(self, traj, input_freq=1000, output_freq=240):
        if len(traj) == 0:
            return np.empty((0, 3), dtype=np.float64)
        x_out, y_out, t_out = self.downsample(
            traj[:, 0], traj[:, 1], traj[:, 2], input_freq=input_freq, output_freq=output_freq
        )
        return self._as_trajectory(x_out, y_out, t_out)

    def build_traj_fixation_index(self, trajectory_xy, scanpath_data):
        traj_xy = np.asarray(trajectory_xy, dtype=np.float64)
        if traj_xy.ndim != 2 or traj_xy.shape[1] != 2:
            raise ValueError(f"trajectory_xy must have shape (N, 2), got {traj_xy.shape}")

        n_traj = traj_xy.shape[0]
        out = np.zeros(n_traj, dtype=np.float32)
        if n_traj == 0:
            return out

        fix_x = np.asarray(scanpath_data.get("X", []), dtype=np.float64)
        fix_y = np.asarray(scanpath_data.get("Y", []), dtype=np.float64)
        if len(fix_x) == 0 or len(fix_y) == 0:
            return out

        n_fix = min(len(fix_x), len(fix_y))
        fix_xy = np.column_stack((fix_x[:n_fix], fix_y[:n_fix]))

        prev_anchor = -1
        for fix_idx, fix_pt in enumerate(fix_xy):
            search_start = prev_anchor + 1
            if search_start >= n_traj:
                break
            dists = np.linalg.norm(traj_xy[search_start:] - fix_pt, axis=1)
            traj_idx = search_start + int(np.argmin(dists))
            out[traj_idx] = float(fix_idx + 1)
            prev_anchor = traj_idx

        out[-1] = float(n_fix + 1)
        anchor_idx = np.flatnonzero(out > 0)
        anchor_vals = out[anchor_idx].astype(np.float32)

        if anchor_idx.size == 0 or anchor_idx[0] != 0:
            anchor_idx = np.insert(anchor_idx, 0, 0)
            anchor_vals = np.insert(anchor_vals, 0, np.float32(0.0))
        if anchor_idx.size < 2:
            return out

        for i in range(anchor_idx.size - 1):
            left = int(anchor_idx[i])
            right = int(anchor_idx[i + 1])
            if right - left <= 1:
                continue
            left_val = float(anchor_vals[i])
            right_val = float(anchor_vals[i + 1])
            # Uniform interpolation of the fixation index between consecutive anchors.
            out[left : right + 1] = np.linspace(left_val, right_val, right - left + 1, dtype=np.float32)
        return out

    def downsample(self, x, y, t, input_freq=1000, output_freq=240):
        assert output_freq < input_freq, "Output frequency must be less than input frequency."
        assert input_freq > 0 and output_freq > 0, "Frequencies must be positive."
        if len(x) == 0:
            return np.array([]), np.array([]), np.array([])

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        if len(x) == 1:
            return x.copy(), y.copy(), t.copy()

        window_ms = 1000.0 / output_freq
        cumulative_time = np.cumsum(t)
        window_indices = (cumulative_time / window_ms).astype(np.int64)

        x_downsampled = []
        y_downsampled = []
        t_downsampled = []
        for window_idx in np.unique(window_indices):
            mask = window_indices == window_idx
            x_downsampled.append(np.mean(x[mask]))
            y_downsampled.append(np.mean(y[mask]))
            t_downsampled.append(np.sum(t[mask]))

        x_out = np.array(x_downsampled)
        y_out = np.array(y_downsampled)
        t_out = np.array(t_downsampled)

        total_input = np.sum(t)
        total_output = np.sum(t_out)
        if not np.isclose(total_input, total_output, rtol=1e-9):
            raise ValueError(
                f"Duration mismatch: input total={total_input:.6f}ms, output total={total_output:.6f}ms"
            )
        return x_out, y_out, t_out

    def cut_trajectory(self, traj, gt_scanpath_xy, gt_total_duration_ms, distance_threshold_px=5.0):
        kept_after_head, discarded_head = self.cut_from_head(
            traj=traj, gt_scanpath_xy=gt_scanpath_xy, distance_threshold_px=distance_threshold_px
        )
        kept_after_tail = kept_after_head
        discarded_tail = np.empty((0, 3), dtype=np.float64)
        if gt_total_duration_ms > 0:
            kept_after_tail, discarded_tail = self.cut_from_tail(
                traj=kept_after_head, max_duration_ms=gt_total_duration_ms, include_partial=False
            )
        return kept_after_tail, discarded_head, discarded_tail

    def cut_from_head(self, traj, gt_scanpath_xy, distance_threshold_px):
        traj = np.asarray(traj, dtype=np.float64)
        gt_scanpath_xy = np.asarray(gt_scanpath_xy, dtype=np.float64)
        if len(traj) == 0 or len(gt_scanpath_xy) == 0:
            return traj, np.empty((0, 3), dtype=np.float64)

        start_point = traj[0, :2]
        start_to_all_gt = np.linalg.norm(gt_scanpath_xy - start_point, axis=1)
        if np.any(start_to_all_gt <= distance_threshold_px):
            return traj, np.empty((0, 3), dtype=np.float64)

        first_gt = gt_scanpath_xy[0]
        dists_to_first_gt = np.linalg.norm(traj[:, :2] - first_gt, axis=1)
        candidate_indices = np.where(dists_to_first_gt <= distance_threshold_px)[0]
        if len(candidate_indices) == 0:
            return traj, np.empty((0, 3), dtype=np.float64)

        cut_idx = int(candidate_indices[0])
        if cut_idx <= 0:
            return traj, np.empty((0, 3), dtype=np.float64)
        return traj[cut_idx:], traj[:cut_idx]

    def cut_from_tail(self, traj, max_duration_ms, include_partial=False):
        assert max_duration_ms > 0, "max_duration_ms must be positive."
        if len(traj) == 0:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

        traj = np.asarray(traj, dtype=np.float64)
        traj_t = traj[:, 2]
        cumulative_time = np.cumsum(traj_t)
        valid_indices = np.where(cumulative_time <= max_duration_ms)[0]

        if len(valid_indices) == 0:
            if include_partial and traj_t[0] > 0:
                clipped = traj[:1].copy()
                clipped[0, 2] = max_duration_ms
                discarded = traj[1:].copy()
                return clipped, discarded
            return np.empty((0, 3), dtype=np.float64), traj.copy()

        last_valid_idx = valid_indices[-1]
        clipped = traj[: last_valid_idx + 1].copy()
        discarded = traj[last_valid_idx + 1 :].copy()
        if include_partial and last_valid_idx < len(traj) - 1:
            remaining_time = max_duration_ms - cumulative_time[last_valid_idx]
            if remaining_time > 0:
                partial_row = traj[last_valid_idx + 1].copy()
                partial_row[2] = remaining_time
                clipped = np.vstack((clipped, partial_row))
                discarded = traj[last_valid_idx + 2 :].copy()
        return clipped, discarded

    def _convert_traj_to_fixation_count(self, traj, original_width, original_height):
        traj = np.asarray(traj, dtype=np.float64)
        if len(traj) == 0:
            return 0
        traj_xy_orig = np.column_stack(
            (
                traj[:, 0] * (original_width / self.frame_width),
                traj[:, 1] * (original_height / self.frame_height),
            )
        )
        _, fix_data, _ = check_fixations(traj_xy_orig)
        converted_xy_orig = np.asarray(fix_data.medianXY, dtype=np.float64)
        return len(converted_xy_orig)

    def _recover_tail_if_scanpath_shorter(
        self,
        kept_traj,
        discarded_tail,
        gt_scanpath_xy,
        original_width,
        original_height,
        distance_threshold_px,
    ):
        kept_traj = np.asarray(kept_traj, dtype=np.float64)
        discarded_tail = np.asarray(discarded_tail, dtype=np.float64)
        gt_scanpath_xy = np.asarray(gt_scanpath_xy, dtype=np.float64)
        if len(kept_traj) == 0 or len(discarded_tail) == 0 or len(gt_scanpath_xy) == 0:
            return kept_traj, discarded_tail

        gt_len = int(len(gt_scanpath_xy))
        converted_len = self._convert_traj_to_fixation_count(
            kept_traj, original_width=original_width, original_height=original_height
        )
        if converted_len >= gt_len:
            return kept_traj, discarded_tail

        last_gt_fix = gt_scanpath_xy[-1]
        tail_xy = discarded_tail[:, :2]
        dists = np.linalg.norm(tail_xy - last_gt_fix, axis=1)
        closest_idx = int(np.argmin(dists))
        add_end = closest_idx + 1
        while add_end < len(discarded_tail):
            dist_to_last = float(np.linalg.norm(discarded_tail[add_end, :2] - last_gt_fix))
            if dist_to_last <= distance_threshold_px:
                add_end += 1
            else:
                break
        recovered = np.vstack((kept_traj, discarded_tail[:add_end]))
        remaining_tail = discarded_tail[add_end:].copy()
        return recovered, remaining_tail
