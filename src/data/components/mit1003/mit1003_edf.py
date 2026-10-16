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


class MIT1003EDFDataset:
    def __init__(
        self,
        name: str,
        root_path: str,
        split: str,
        task_embeddings_path: str,
        threshold: float = -1.0,
        threshold_no_dur: float = -1.0,
        use_cache: bool = True,
        sample_rate: int = 240,
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
        self.cache_path = Path(self.root_path) / "mit1003_edf_cache.pkl"
        self.subject_id_to_code = {
            idx + 1: code
            for idx, code in enumerate(
                # Official MIT1003 observer order from DatabaseCode/README.txt
                # is: CNG, ajs, emb, ems, ff, hp, jcw, jw, kae, krl, po, tmj, tu, ya, zb
                ["CNG", "ajs", "emb", "ems", "ff", "hp", "jcw", "jw", "kae", "krl", "po", "tmj", "tu", "ya", "zb"]
            )
        }

        json_split = "validation" if split == "valid" else split
        with open(Path(self.root_path, f"mit1003_fixations_{json_split}.json"), "rb") as f:
            loaded = json.load(f)

        samples = [s for s in loaded if s.get("split", json_split) == json_split]
        if max_samples > 0:
            samples = samples[:max_samples]

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

        traj_data = self._load_or_build_traj_cache(use_cache=use_cache)
        traj_index = self._index_trajectories_by_img_subject(traj_data)

        joined_samples = []
        missing_traj_keys = []
        dropped_invalid_count = 0
        for sample in tqdm(self.samples, desc="Processing MIT1003 EDF samples"):
            subject_code = self._subject_as_is(sample.get("subject", ""))
            match_key = self._match_sample_to_traj_key(sample, subject_code, traj_index)
            if match_key is None:
                missing_traj_keys.append((sample["name"], sample.get("subject", None), subject_code))
                continue

            trajectory = traj_index[match_key]
            processed = self._attach_processed_trajectory(sample.copy(), trajectory)
            invalid_stats = self._trajectory_invalid_stats(processed)
            if invalid_stats is not None:
                dropped_invalid_count += 1
                continue
            joined_samples.append(processed)

        if missing_traj_keys:
            unique_missing = sorted(set(missing_traj_keys))
            preview = ", ".join([str(k) for k in unique_missing[:10]])
            raise ValueError(
                "Some fixation samples do not have corresponding MIT EDF trajectories. "
                f"Missing: {len(missing_traj_keys)} / {len(self.samples)} samples. "
                f"Unique missing keys: {len(unique_missing)}. "
                f"Examples: {preview}"
            )

        print(
            "MIT1003 EDF filtering summary "
            f"[split={json_split}]: dropped_invalid_after_truncate={dropped_invalid_count}, "
            f"kept={len(joined_samples)}, total_fixation_samples={len(self.samples)}"
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

        original_width = float(sample["width"])
        original_height = float(sample["height"])
        img_features_path = Path(self.root_path, self.img_features_dir, Path(img_filename).stem + ".pth")
        img_feats = torch.load(img_features_path).unsqueeze(0)

        # Normalize coordinates by the image size and convert durations from ms to s.
        trajectory = np.stack((sample["X_traj"], sample["Y_traj"], sample["T_traj"]), axis=1).astype(np.float32)
        trajectory[:, 0] /= original_width
        trajectory[:, 1] /= original_height
        trajectory = torch.from_numpy(trajectory).float()
        trajectory[:, 2] /= 1000.0

        scanpath = np.stack(
            (np.asarray(sample["X"], dtype=np.float32), np.asarray(sample["Y"], dtype=np.float32), np.asarray(sample["T"], dtype=np.float32)),
            axis=1,
        ).astype(np.float32)
        scanpath[:, 0] /= original_width
        scanpath[:, 1] /= original_height
        scanpath = torch.from_numpy(scanpath).float()
        scanpath[:, 2] /= 1000.0

        return {
            "img_filename": img_filename,
            "original_img_size": (int(original_width), int(original_height)),
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

    def _trajectory_invalid_stats(self, sample):
        x = np.asarray(sample.get("X_traj", []), dtype=np.float64)
        y = np.asarray(sample.get("Y_traj", []), dtype=np.float64)
        if x.shape != y.shape:
            # Treat malformed trajectories as invalid.
            return {"nan": 0, "negative": 0, "out_of_frame": int(max(len(x), len(y)))}
        if x.size == 0:
            return None

        width = float(sample.get("width", np.nan))
        height = float(sample.get("height", np.nan))
        if np.isnan(width) or np.isnan(height):
            return {"nan": 0, "negative": 0, "out_of_frame": int(x.size)}

        nan_mask = np.isnan(x) | np.isnan(y)
        negative_mask = (x < 0) | (y < 0)
        out_of_frame_mask = (x > width) | (y > height)

        nan_count = int(nan_mask.sum())
        negative_count = int(negative_mask.sum())
        out_of_frame_count = int(out_of_frame_mask.sum())

        if nan_count == 0 and negative_count == 0 and out_of_frame_count == 0:
            return None
        return {
            "nan": nan_count,
            "negative": negative_count,
            "out_of_frame": out_of_frame_count,
        }

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

        data_dir = Path(self.root_path) / "DATA"
        if not data_dir.exists():
            raise FileNotFoundError(f"MIT trajectory DATA directory not found: {data_dir}")

        all_data = []
        subject_dirs = [p for p in data_dir.iterdir() if p.is_dir()]
        for subject_dir in tqdm(sorted(subject_dirs), desc="Loading MIT1003 EDF MAT files"):
            for mat_path in subject_dir.glob("*.mat"):
                try:
                    traj = self._load_traj_mfile(str(mat_path))
                    traj = self._filter_invalid_samples(traj)
                    all_data.append(
                        {
                            "img_name": mat_path.stem,
                            "subject": subject_dir.name,
                            "x": traj[:, 0].astype(np.float64).tolist(),
                            "y": traj[:, 1].astype(np.float64).tolist(),
                            "t": np.arange(len(traj), dtype=np.float64).tolist(),
                        }
                    )
                except Exception as e:
                    print(f"Error reading {mat_path}: {e}")
                    traceback.print_exc()

        with open(self.cache_path, "wb") as f:
            pickle.dump(all_data, f)
        return all_data

    @staticmethod
    def _load_traj_mfile(traj_pth: str) -> np.ndarray:
        try:
            from scipy import io
        except Exception as e:
            raise ImportError(
                "scipy is required to load MIT .mat trajectory files. Install scipy in your runtime environment."
            ) from e

        traj_mat = io.loadmat(traj_pth)
        traj = traj_mat[list(traj_mat.keys())[-1]][0, 0][-1][0, 0][-1]
        if traj.shape == (1, 1):
            traj = traj_mat[list(traj_mat.keys())[-1]][0, 0][-1][0, 0][2][:, :2]
        traj = np.asarray(traj, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] < 2:
            raise ValueError(f"Unexpected trajectory shape in {traj_pth}: {traj.shape}")
        return traj[:, :2]

    @staticmethod
    def _filter_invalid_samples(traj_xy: np.ndarray) -> np.ndarray:
        traj_xy = np.asarray(traj_xy, dtype=np.float64)
        if traj_xy.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        traj_xy = traj_xy[~np.isnan(traj_xy).any(axis=1)]
        traj_xy = traj_xy[(traj_xy >= 0).all(axis=1)]
        return traj_xy

    @staticmethod
    def _normalize_subject(subject):
        return str(subject).strip()

    @staticmethod
    def _image_key_variants(img_name):
        p = Path(str(img_name))
        stem = p.stem.lower()
        base = p.name.lower()
        return {base, stem, f"{stem}.jpg", f"{stem}.jpeg", f"{stem}.png"}

    def _subject_as_is(self, subject):
        if isinstance(subject, str):
            out = subject.strip()
            if out == "":
                raise ValueError("MIT1003 EDF subject ID cannot be empty.")
            return out

        try:
            subject_id = int(subject)
        except Exception as e:
            raise ValueError(
                f"Unsupported MIT1003 subject value {subject!r} (type={type(subject).__name__}). "
                "Expected subject code string (e.g. 'CNG') or numeric id in [1..15]."
            ) from e

        if subject_id not in self.subject_id_to_code:
            raise ValueError(
                f"Unknown MIT1003 numeric subject id '{subject_id}'. "
                f"Available ids: {sorted(self.subject_id_to_code)}"
            )
        return self.subject_id_to_code[subject_id]

    def _index_trajectories_by_img_subject(self, trajectories):
        indexed = {}
        duplicate = []
        for d in trajectories:
            subject_code = str(d.get("subject", "")).strip()
            for img_variant in self._image_key_variants(d.get("img_name", "")):
                key = (img_variant, subject_code)
                if key in indexed:
                    duplicate.append(key)
                    continue
                indexed[key] = d
        if duplicate:
            uniq = sorted(set(duplicate))
            preview = ", ".join([str(k) for k in uniq[:10]])
            raise ValueError(
                "Found duplicate MIT EDF trajectory keys for (image, subject). "
                f"Unique duplicates: {len(uniq)}. Examples: {preview}"
            )
        return indexed

    def _match_sample_to_traj_key(self, sample, subject_code, traj_index):
        for img_variant in self._image_key_variants(sample["name"]):
            key = (img_variant, subject_code)
            if key in traj_index:
                return key
        return None

    def _attach_processed_trajectory(self, sample, trajectory):
        sample_rate = self.sample_rate
        unit = 1 / sample_rate * 1000.0

        raw_t = np.asarray(trajectory["t"], dtype=np.float64)
        if len(raw_t) > 0:
            raw_t = np.array([unit] + (unit * (raw_t[1:] - raw_t[:-1])).tolist(), dtype=np.float64)
        raw_x = np.asarray(trajectory["x"], dtype=np.float64)
        raw_y = np.asarray(trajectory["y"], dtype=np.float64)
        raw_traj = self._as_trajectory(raw_x, raw_y, raw_t)
        raw_traj_uncut = raw_traj

        gt_scanpath_xy = np.column_stack((np.asarray(sample["X"], dtype=np.float64), np.asarray(sample["Y"], dtype=np.float64)))
        gt_total_duration_ms = float(np.sum(np.asarray(sample["T"], dtype=np.float64)))
        # Trim the trajectory to the span covered by the human scanpath (already sampled at 240 Hz).
        raw_traj, _, raw_traj_discarded_tail = self.cut_trajectory(
            traj=raw_traj,
            gt_scanpath_xy=gt_scanpath_xy,
            gt_total_duration_ms=gt_total_duration_ms,
            distance_threshold_px=self.head_align_distance_threshold_px,
        )

        if sample_rate > 240:
            traj = self.downsample_trajectory(raw_traj, input_freq=sample_rate, output_freq=240)
            discarded_tail = self.downsample_trajectory(raw_traj_discarded_tail, input_freq=sample_rate, output_freq=240)
            if len(traj) == 0:
                traj = self.downsample_trajectory(raw_traj_uncut, input_freq=sample_rate, output_freq=240)
                discarded_tail = np.empty((0, 3), dtype=np.float64)
        else:
            traj = raw_traj
            discarded_tail = raw_traj_discarded_tail
            if len(traj) == 0:
                traj = raw_traj_uncut
                discarded_tail = np.empty((0, 3), dtype=np.float64)

        traj, discarded_tail = self._recover_tail_if_scanpath_shorter(
            kept_traj=traj,
            discarded_tail=discarded_tail,
            gt_scanpath_xy=gt_scanpath_xy,
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

    @staticmethod
    def _convert_traj_to_fixation_count(traj):
        traj = np.asarray(traj, dtype=np.float64)
        if len(traj) == 0:
            return 0
        _, fix_data, _ = check_fixations(traj[:, :2])
        converted_xy = np.asarray(fix_data.medianXY, dtype=np.float64)
        return len(converted_xy)

    def _recover_tail_if_scanpath_shorter(self, kept_traj, discarded_tail, gt_scanpath_xy, distance_threshold_px):
        kept_traj = np.asarray(kept_traj, dtype=np.float64)
        discarded_tail = np.asarray(discarded_tail, dtype=np.float64)
        gt_scanpath_xy = np.asarray(gt_scanpath_xy, dtype=np.float64)
        if len(kept_traj) == 0 or len(discarded_tail) == 0 or len(gt_scanpath_xy) == 0:
            return kept_traj, discarded_tail

        gt_len = int(len(gt_scanpath_xy))
        converted_len = self._convert_traj_to_fixation_count(kept_traj)
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
