import torch
from collections import Counter
from pathlib import Path
from PIL import Image
import numpy as np
import json
import pickle
from tqdm import tqdm
import traceback

from src.utils.load_edf import read_edfasc
from src.utils.check_fixations import check_fixations


class COCOSearch18EDFTargetAbsentDataset:
    def __init__(self, name: str, root_path: str, split: str, task_embeddings_path: str, threshold: float = -1.,
                 threshold_no_dur: float = -1., use_cache: bool = True, sample_rate: int = 1000,
                 img_features_dir: str = 'dinov2_base_timm_image_features_ta', max_samples: int = -1,
                 head_align_distance_threshold_px: float = 10.0) -> None:
        self.name = name
        self.root_path = root_path
        self.use_cache = use_cache
        # Sequence-score similarity thresholds used by the evaluation metrics.
        self.threshold = threshold
        self.threshold_no_dur = threshold_no_dur
        self.cache_path = Path(root_path) / 'cocosearch18_edf_cache.pkl'
        self.head_align_distance_threshold_px = head_align_distance_threshold_px
        if split == 'valid':
            split = 'validation'

        def normalize_task_key(task_name):
            return str(task_name).replace(' ', '_')

        # Using the already resized scanpath data provided by ScanDiff
        if split in ['train', 'validation']:
            with open(Path(self.root_path, 'coco_search18_fixations_TA_trainval.json'), 'rb') as f:
                self.samples = json.load(f)

            self.samples = [s for s in self.samples if s.get('split') == split]
        else:
            with open(Path(self.root_path, 'coco_search18_fixations_TA_test.json'), 'rb') as f:
                self.samples = json.load(f)

        sample_keys = [
            (s['name'], s['subject'], normalize_task_key(s['task']))
            for s in self.samples
        ]
        duplicate_keys = [(k, c) for k, c in Counter(sample_keys).items() if c > 1]
        if duplicate_keys:
            duplicate_preview = ", ".join([f"{k} x{c}" for k, c in duplicate_keys[:10]])
            raise ValueError(
                "Found duplicate fixation keys for (name, subject, task). "
                f"Count: {len(duplicate_keys)}. Examples: {duplicate_preview}"
            )

        self.samples = sorted(
            self.samples,
            key=lambda x: (x['name'], x['subject'], normalize_task_key(x['task']))
        )
        if max_samples > 0:
            self.samples = self.samples[:max_samples]

        # Load edf trajectory
        if use_cache and self.cache_path.exists():
            traj_data = pickle.load(open(self.cache_path, 'rb'))
        else:
            all_data = []
            all_asc_files = list(Path(self.root_path).rglob('*.asc'))
            for file_path in tqdm(all_asc_files, desc="Loading EDF ASC files"):
                try:
                    data = read_edfasc(file_path)
                    all_data.extend(data)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    traceback.print_exc()
            all_data_dict = {}
            for entry in all_data:
                key = (
                    entry['img_name'],
                    entry['subject'],
                    normalize_task_key(entry['category']),
                )
                all_data_dict[key] = entry
            traj_data = all_data_dict
            with open(self.cache_path, 'wb') as f:
                pickle.dump(traj_data, f)

        # Add trajectory data to samples
        traj_samples = []
        missing_traj_keys = []
        for sample in tqdm(self.samples, desc="Processing samples with trajectory data"):
            key = (
                sample['name'],
                sample['subject'],
                normalize_task_key(sample['task']),
            )
            if key in traj_data:
                trajectory = traj_data[key]
                unit = 1 / sample_rate * 1000  # unit time in ms
                trajectory['t'] = [unit] + (unit * (np.array(trajectory['t'][1:]) - np.array(trajectory['t'][:-1]))).tolist()
                raw_x = np.asarray(trajectory['x'], dtype=np.float64)
                raw_y = np.asarray(trajectory['y'], dtype=np.float64)
                raw_t = np.asarray(trajectory['t'], dtype=np.float64)

                # Rescale raw trajectory from original image size to 1680x1050 to align with TA scanpath data.
                task_folder = str(sample['task']).replace('_', ' ')
                original_width, original_height = Image.open(Path(self.root_path, 'images', task_folder, sample['name'])).size
                raw_x = raw_x * (1680 / original_width)
                raw_y = raw_y * (1050 / original_height)
                raw_traj = self._as_trajectory(raw_x, raw_y, raw_t)
                raw_traj_uncut = raw_traj
                gt_scanpath_xy = np.column_stack(
                    (
                        np.asarray(sample['X'], dtype=np.float64),
                        np.asarray(sample['Y'], dtype=np.float64),
                    )
                )

                gt_total_duration_ms = float(np.sum(np.asarray(sample['T'], dtype=np.float64)))
                (
                    raw_traj,
                    raw_traj_discarded_head,
                    raw_traj_discarded_tail,
                ) = self.cut_trajectory(
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

                # If processed trajectory yields fewer fixations than the loaded GT scanpath,
                # selectively add back tail segments (TA-only recovery).
                traj, discarded_tail = self._recover_tail_if_scanpath_shorter(
                    kept_traj=traj,
                    discarded_tail=discarded_tail,
                    gt_scanpath_xy=gt_scanpath_xy,
                    original_width=original_width,
                    original_height=original_height,
                    distance_threshold_px=self.head_align_distance_threshold_px,
                )

                x, y, t = self._split_trajectory(traj)
                sample['X_traj'] = x
                sample['Y_traj'] = y
                sample['T_traj'] = t
                sample['traj_fixation_index'] = self.build_traj_fixation_index(
                    trajectory_xy=np.column_stack((x, y)),
                    scanpath_data=sample,
                )
                traj_samples.append(sample)
            else:
                missing_traj_keys.append(key)

        if missing_traj_keys:
            unique_missing = sorted(set(missing_traj_keys))
            preview = ", ".join([str(k) for k in unique_missing[:10]])
            raise ValueError(
                "Some fixation samples do not have corresponding EDF trajectories. "
                f"Missing: {len(missing_traj_keys)} / {len(self.samples)} samples. "
                f"Unique missing keys: {len(unique_missing)}. "
                f"Examples: {preview}"
            )
        self.samples = traj_samples

        self.task_embeddings = np.load(task_embeddings_path, allow_pickle=True).item()

        self.img_features_dir = img_features_dir

        self.subject_per_img_per_task = {}

        for s in self.samples:
            task_name = normalize_task_key(s['task'])

            if f'{s["name"]}_{task_name}' not in self.subject_per_img_per_task:
                self.subject_per_img_per_task[f'{s["name"]}_{task_name}'] = 1
            else:
                self.subject_per_img_per_task[f'{s["name"]}_{task_name}'] += 1

    def __getitem__(self, index: int):
        sample = self.samples[index]
        img_filename = sample['name']
        task_string = sample['task']

        task_embedding = torch.from_numpy(self.task_embeddings[task_string])

        task_string = task_string.replace(' ', '_')
        task_folder = task_string.replace('_', ' ')

        num_viewers = self.subject_per_img_per_task[f'{img_filename}_{task_string}']
        original_img_size = Image.open(Path(self.root_path, 'images', task_folder, Path(img_filename))).size
        img_features_path = Path(self.root_path, self.img_features_dir, task_string, Path(img_filename).stem + '.pth')
        img_feats = torch.load(img_features_path).unsqueeze(0)

        num_points = len(sample['X_traj'])
        x = sample['X_traj']
        y = sample['Y_traj']
        t = sample['T_traj'][:num_points]

        trajectory = np.stack((x, y, t)).T

        trajectory[:, 0] /= 1680  # TA annotations are in the 1680x1050 screen frame
        trajectory[:, 1] /= 1050

        trajectory = torch.from_numpy(trajectory).float()

        trajectory[:, 2] /= 1000.0  # ms -> s

        num_fixations = len(sample['X'])
        fix_x = np.asarray(sample['X'][:num_fixations], dtype=np.float32)
        fix_y = np.asarray(sample['Y'][:num_fixations], dtype=np.float32)
        fix_t = np.asarray(sample['T'][:num_fixations], dtype=np.float32)
        scanpath = np.stack((fix_x, fix_y, fix_t)).T

        scanpath[:, 0] /= 1680
        scanpath[:, 1] /= 1050

        scanpath = torch.from_numpy(scanpath).float()

        scanpath[:, 2] /= 1000.0  # ms -> s

        return {'img_filename': img_filename, 'original_img_size': original_img_size, 'img': img_feats.squeeze(), 'scanpath': scanpath, 'task': task_string,
                'task_embedding': task_embedding, 'num_viewers': num_viewers, 'dataset': self.name, 'question_id': "",
                'trajectory': trajectory,
                'traj_fixation_index': torch.from_numpy(np.asarray(sample['traj_fixation_index'], dtype=np.float32))}

    def __len__(self) -> int:
        return len(self.samples)

    def downsample(self, x, y, t, input_freq=1000, output_freq=240):
        """
        Downsample a scanpath sequence from high frequency (e.g., 1000Hz) to lower frequency (e.g., 240Hz).

        This method handles:
        - Temporal aggregation: Groups consecutive samples that fall within the same output time window
        - Spatial averaging: Computes mean x,y for samples in the same window
        - Duration accumulation: Sums durations within each window to preserve total time
        - Edge cases: Empty sequences, single samples, mismatched lengths

        Duration handling example:
        - Input at 1000Hz: t=[1, 1, 1, 1] ms (4 samples, 4ms total)
        - Downsample to 500Hz (window=2ms): aggregates [1,1] + [1,1] -> t=[2, 2] ms (2 samples, 4ms total)
        - Downsample to 250Hz (window=4ms): aggregates [1,1,1,1] -> t=[4] ms (1 sample, 4ms total)

        Args:
            x (list/array): X coordinates (normalized or absolute)
            y (list/array): Y coordinates (normalized or absolute)
            t (list/array): Durations in milliseconds for each sample
            input_freq (int): Original sampling frequency in Hz (default: 1000Hz)
            output_freq (int): Target sampling frequency in Hz (default: 240Hz)

        Returns:
            tuple: (x_downsampled, y_downsampled, t_downsampled) as numpy arrays
        """
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

        unique_windows = np.unique(window_indices)

        for window_idx in unique_windows:
            mask = window_indices == window_idx

            x_mean = np.mean(x[mask])
            y_mean = np.mean(y[mask])
            t_sum = np.sum(t[mask])

            x_downsampled.append(x_mean)
            y_downsampled.append(y_mean)
            t_downsampled.append(t_sum)

        x_out = np.array(x_downsampled)
        y_out = np.array(y_downsampled)
        t_out = np.array(t_downsampled)

        total_input = np.sum(t)
        total_output = np.sum(t_out)
        if not np.isclose(total_input, total_output, rtol=1e-9):
            raise ValueError(f"Duration mismatch: input total={total_input:.6f}ms, output total={total_output:.6f}ms")

        return x_out, y_out, t_out

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
        x_out, y_out, t_out = self.downsample(traj[:, 0], traj[:, 1], traj[:, 2], input_freq=input_freq, output_freq=output_freq)
        return self._as_trajectory(x_out, y_out, t_out)

    def build_traj_fixation_index(self, trajectory_xy, scanpath_data):
        traj_xy = np.asarray(trajectory_xy, dtype=np.float64)
        if traj_xy.ndim != 2 or traj_xy.shape[1] != 2:
            raise ValueError(f"trajectory_xy must have shape (N, 2), got {traj_xy.shape}")

        n_traj = traj_xy.shape[0]
        out = np.zeros(n_traj, dtype=np.float32)
        if n_traj == 0:
            return out

        fix_x = np.asarray(scanpath_data.get('X', []), dtype=np.float64)
        fix_y = np.asarray(scanpath_data.get('Y', []), dtype=np.float64)
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
            out[left:right + 1] = np.linspace(left_val, right_val, right - left + 1, dtype=np.float32)

        return out

    def cut_trajectory(
        self,
        traj,
        gt_scanpath_xy,
        gt_total_duration_ms,
        distance_threshold_px=5.0,
    ):
        kept_after_head, discarded_head = self.cut_from_head(
            traj=traj,
            gt_scanpath_xy=gt_scanpath_xy,
            distance_threshold_px=distance_threshold_px,
        )

        kept_after_tail = kept_after_head
        discarded_tail = np.empty((0, 3), dtype=np.float64)
        if gt_total_duration_ms > 0:
            kept_after_tail, discarded_tail = self.cut_from_tail(
                traj=kept_after_head,
                max_duration_ms=gt_total_duration_ms,
                include_partial=False,
            )

        return kept_after_tail, discarded_head, discarded_tail

    def cut_from_head(
        self,
        traj,
        gt_scanpath_xy,
        distance_threshold_px,
    ):
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

    def cut_from_tail(
        self,
        traj,
        max_duration_ms,
        include_partial=False,
    ):
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
            else:
                return np.empty((0, 3), dtype=np.float64), traj.copy()

        last_valid_idx = valid_indices[-1]
        clipped = traj[:last_valid_idx + 1].copy()
        discarded = traj[last_valid_idx + 1:].copy()

        if include_partial and last_valid_idx < len(traj) - 1:
            remaining_time = max_duration_ms - cumulative_time[last_valid_idx]
            if remaining_time > 0:
                partial_row = traj[last_valid_idx + 1].copy()
                partial_row[2] = remaining_time
                clipped = np.vstack((clipped, partial_row))
                discarded = traj[last_valid_idx + 2:].copy()

        return clipped, discarded

    def _convert_traj_to_fixation_count(self, traj_ta, original_width, original_height):
        """Convert TA-frame trajectory to fixation count via check_fixations."""
        traj_ta = np.asarray(traj_ta, dtype=np.float64)
        if len(traj_ta) == 0:
            return 0
        traj_xy_orig = np.column_stack(
            (
                traj_ta[:, 0] * (original_width / 1680.0),
                traj_ta[:, 1] * (original_height / 1050.0),
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
        """
        Recover part of discarded tail when converted trajectory scanpath is shorter than loaded GT scanpath.

        Rule:
        1) Compare converted fixation count from kept_traj with loaded GT count.
        2) If kept is shorter, find the closest point in discarded tail to the last GT fixation.
        3) Add back discarded tail points up to that closest point.
        4) Continue adding subsequent points while they remain within distance_threshold_px
           to the last GT fixation (in TA coordinates).
        """
        kept_traj = np.asarray(kept_traj, dtype=np.float64)
        discarded_tail = np.asarray(discarded_tail, dtype=np.float64)
        gt_scanpath_xy = np.asarray(gt_scanpath_xy, dtype=np.float64)

        if len(kept_traj) == 0 or len(discarded_tail) == 0 or len(gt_scanpath_xy) == 0:
            return kept_traj, discarded_tail

        gt_len = int(len(gt_scanpath_xy))
        converted_len = self._convert_traj_to_fixation_count(
            kept_traj,
            original_width=original_width,
            original_height=original_height,
        )
        if converted_len >= gt_len:
            return kept_traj, discarded_tail

        last_gt_fix = gt_scanpath_xy[-1]
        tail_xy = discarded_tail[:, :2]
        dists = np.linalg.norm(tail_xy - last_gt_fix, axis=1)
        closest_idx = int(np.argmin(dists))

        # Add points up to and including the closest point to the last GT fixation.
        add_end = closest_idx + 1

        # From that point onward, keep adding while still within threshold.
        while add_end < len(discarded_tail):
            dist_to_last = float(np.linalg.norm(discarded_tail[add_end, :2] - last_gt_fix))
            if dist_to_last <= distance_threshold_px:
                add_end += 1
            else:
                break

        recovered = np.vstack((kept_traj, discarded_tail[:add_end]))
        remaining_tail = discarded_tail[add_end:].copy()
        return recovered, remaining_tail
