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


class COCOSearch18EDFTargetPresentDataset:
    def __init__(self, name: str, root_path: str, split: str, task_embeddings_path: str, threshold: float = -1.,
                 threshold_no_dur: float = -1., use_cache: bool = True, sample_rate: int = 1000,
                 img_features_dir: str = 'dinov2_base_timm_image_features_tp', img_dir: str = 'images_tp',
                 max_samples: int = -1,
                 head_align_distance_threshold_px: float = 10.0) -> None:
        self.name = name
        self.root_path = root_path
        self.use_cache = use_cache
        self.img_dir = img_dir
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
        with open(Path(self.root_path, f'coco_search18_fixations_TP_{split}.json'), 'rb') as f:
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
        for sample_idx, sample in enumerate(tqdm(self.samples, desc="Processing samples with trajectory data")):
            key = (
                sample['name'],
                sample['subject'],
                normalize_task_key(sample['task']),
            )
            if key in traj_data:
                trajectory = traj_data[key]
                unit = 1 / sample_rate * 1000  # unit time in ms
                trajectory['t'] = [unit] + (unit * (np.array(trajectory['t'][1:]) - np.array(trajectory['t'][:-1]))).tolist()  # 1000 Hz sample rate for raw trajectory

                # Rescale raw trajectory to 512x320 frame to align with provided GT scanpath.
                raw_x = np.asarray(trajectory['x'], dtype=np.float64)
                raw_y = np.asarray(trajectory['y'], dtype=np.float64)
                raw_t = np.asarray(trajectory['t'], dtype=np.float64)
                task_folder = normalize_task_key(sample['task'])
                original_width, original_height = Image.open(Path(self.root_path, self.img_dir, task_folder, sample['name'])).size
                raw_x = raw_x * (512 / original_width)
                raw_y = raw_y * (320 / original_height)
                raw_traj = self._as_trajectory(raw_x, raw_y, raw_t)
                raw_traj_uncut = raw_traj
                gt_scanpath_xy = np.column_stack(
                    (
                        np.asarray(sample['X'], dtype=np.float64),
                        np.asarray(sample['Y'], dtype=np.float64),
                    )
                )

                # Cut raw trajectory from head (spatial alignment) and tail (GT duration) before downsampling.
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
                # selectively add back tail segments.
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
        
        self.subject_per_img_per_task = {} # this variable also contains the number of unique images of that split
        
        for s in self.samples:
            task = s['task']
            if task == 'potted plant':
                task = 'potted_plant'
            
            if task == 'stop sign':
                task = 'stop_sign'
            
            if f'{s["name"]}_{task}' not in self.subject_per_img_per_task: # the number is not the same for all imgs because too short scanpaths are discarded.
                self.subject_per_img_per_task[f'{s["name"]}_{task}'] = 1
            else:
                self.subject_per_img_per_task[f'{s["name"]}_{task}'] += 1
    
    def __getitem__(self, index: int):
        sample = self.samples[index]
        img_filename = sample['name']
        task_string = sample['task']
        
        task_embedding = torch.from_numpy(self.task_embeddings[task_string]) # embedding extracted with roberta
        
        # treat specific edge cases
        if task_string == 'potted plant':
            task_string = 'potted_plant'
            
        if task_string == 'stop sign':
            task_string = 'stop_sign'
        
        num_viewers = self.subject_per_img_per_task[f'{img_filename}_{task_string}']
        original_img_size = Image.open(Path(self.root_path, self.img_dir, task_string, Path(img_filename))).size
        img_features_path = Path(self.root_path, self.img_features_dir, task_string, Path(img_filename).stem + '.pth')
        img_feats = torch.load(img_features_path).unsqueeze(0)
        
        #if self.split == 'train' or self.split == 'val':
        num_points = len(sample['X_traj'])
        x = sample['X_traj']
        y = sample['Y_traj']
        t = sample['T_traj'][:num_points]
        
        trajectory = np.stack((x,y,t)).T # coords are in range [512x320]
        #coords = (coords + 1) / 2 #put in range [0,1]
        
        trajectory[:,0] /= 512
        trajectory[:,1] /= 320

        
        #durations = np.hstack((sample['arrival_times'], sample['t_end']))[1:] - sample['arrival_times']
        #durations = np.reshape(durations, (-1, 1))
        #scanpath = np.hstack((coords, durations))
        trajectory = torch.from_numpy(trajectory).float() # gt duration is in seconds
        
        trajectory[:,2] /= 1000.0  # ms -> s

        num_fixations = len(sample['X'])
        fix_x = np.asarray(sample['X'][:num_fixations], dtype=np.float32)
        fix_y = np.asarray(sample['Y'][:num_fixations], dtype=np.float32)
        fix_t = np.asarray(sample['T'][:num_fixations], dtype=np.float32)
        scanpath = np.stack((fix_x, fix_y, fix_t)).T

        scanpath[:, 0] /= 512
        scanpath[:, 1] /= 320

        scanpath = torch.from_numpy(scanpath).float()

        scanpath[:, 2] /= 1000.0  # ms -> s

        return {'img_filename': img_filename, 'original_img_size': original_img_size, 'img': img_feats.squeeze(), 'scanpath': scanpath, 'task': task_string, 
                'task_embedding': task_embedding, 'num_viewers': num_viewers, 'dataset': self.name, 'question_id': "",
                'trajectory': trajectory,
                'traj_fixation_index': torch.from_numpy(np.asarray(sample['traj_fixation_index'], dtype=np.float32))}

    def __len__(self) -> int:
        return len(self.samples)

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
        """
        Build a 1D array that maps each trajectory point to a fixation index.

        Steps:
        1) Initialize a zero array of length len(trajectory_xy)
        2) For each fixation in temporal order, find the closest trajectory point
           constrained to points strictly after the previous anchor
        3) For each interval between assigned points, linearly interpolate indices

        Args:
            trajectory_xy: array-like of shape (N, 2) for processed trajectory x/y.
            scanpath_data: loaded scanpath record containing 'X' and 'Y'.

        Returns:
            np.ndarray of shape (N,), dtype float32.
        """
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

        # Monotonic anchoring: each next fixation can only match points after the
        # previous anchor so anchor indices are strictly increasing in time.
        prev_anchor = -1
        for fix_idx, fix_pt in enumerate(fix_xy):
            search_start = prev_anchor + 1
            if search_start >= n_traj:
                break
            dists = np.linalg.norm(traj_xy[search_start:] - fix_pt, axis=1)
            traj_idx = search_start + int(np.argmin(dists))
            out[traj_idx] = float(fix_idx + 1)  # 1-based fixation index
            prev_anchor = traj_idx

        # Add a terminal index at the end of the trajectory.
        out[-1] = float(n_fix + 1)

        anchor_idx = np.flatnonzero(out > 0)
        anchor_vals = out[anchor_idx].astype(np.float32)

        # Ensure the prefix is also interpolated by adding an explicit start anchor.
        if anchor_idx.size == 0 or anchor_idx[0] != 0:
            anchor_idx = np.insert(anchor_idx, 0, 0)
            anchor_vals = np.insert(anchor_vals, 0, np.float32(0.0))

        if anchor_idx.size < 2:
            return out

        # Interpolate between anchor intervals.
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
        - Downsample to 500Hz (window=2ms): aggregates [1,1] + [1,1] → t=[2, 2] ms (2 samples, 4ms total)
        - Downsample to 250Hz (window=4ms): aggregates [1,1,1,1] → t=[4] ms (1 sample, 4ms total)
        
        Args:
            x (list/array): X coordinates (normalized or absolute)
            y (list/array): Y coordinates (normalized or absolute)
            t (list/array): Durations in milliseconds for each sample
            input_freq (int): Original sampling frequency in Hz (default: 1000Hz)
            output_freq (int): Target sampling frequency in Hz (default: 240Hz)
            
        Returns:
            tuple: (x_downsampled, y_downsampled, t_downsampled) as numpy arrays
        """
        # Input validation
        assert output_freq < input_freq, "Output frequency must be less than input frequency."
        assert input_freq > 0 and output_freq > 0, "Frequencies must be positive."
        
        # Handle empty input
        if len(x) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Convert to numpy arrays for robust processing
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        
        # Handle single sample
        if len(x) == 1:
            return x.copy(), y.copy(), t.copy()
        
        # Calculate the temporal window size in milliseconds
        window_ms = 1000.0 / output_freq  # e.g., 1000/240 ≈ 4.17ms per window
        
        # Compute cumulative time to assign samples to temporal windows
        cumulative_time = np.cumsum(t)
        
        # Assign each sample to a window index based on cumulative time
        # Add small epsilon to handle floating point edge cases
        window_indices = (cumulative_time / window_ms).astype(np.int64)
        
        # Initialize output lists
        x_downsampled = []
        y_downsampled = []
        t_downsampled = []
        
        # Group samples by window and aggregate
        unique_windows = np.unique(window_indices)
        
        for window_idx in unique_windows:
            # Get all samples in this temporal window
            mask = window_indices == window_idx
            
            # Aggregate spatial coordinates (mean)
            x_mean = np.mean(x[mask])
            y_mean = np.mean(y[mask])
            
            # Aggregate durations (sum) - this properly handles sampling rate conversion
            # E.g., at 1000Hz with t=[1,1], downsampling to 500Hz gives t=2
            t_sum = np.sum(t[mask])
            
            x_downsampled.append(x_mean)
            y_downsampled.append(y_mean)
            t_downsampled.append(t_sum)
        
        # Convert to arrays
        x_out = np.array(x_downsampled)
        y_out = np.array(y_downsampled)
        t_out = np.array(t_downsampled)
        
        # Validation: total duration should be preserved (within floating point tolerance)
        total_input = np.sum(t)
        total_output = np.sum(t_out)
        if not np.isclose(total_input, total_output, rtol=1e-9):
            raise ValueError(f"Duration mismatch: input total={total_input:.6f}ms, output total={total_output:.6f}ms")
        
        return x_out, y_out, t_out

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

        # Return head/tail discarded parts separately so they can be plotted independently.
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

        # If trajectory already starts near any GT fixation, do not cut the head.
        if np.any(start_to_all_gt <= distance_threshold_px):
            return traj, np.empty((0, 3), dtype=np.float64)

        # Otherwise, find the first trajectory sample close to the first GT fixation.
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
        """
        Cut the trajectory tail to a maximum duration in milliseconds.
        Drops all points after the cumulative duration exceeds max_duration_ms.
        
        Args:
            traj_x (list/array): X coordinates of the trajectory
            traj_y (list/array): Y coordinates of the trajectory
            traj_t (list/array): Durations in milliseconds for each sample
            max_duration_ms (float): Maximum allowed duration in milliseconds
            include_partial (bool): If True, includes the last point with adjusted duration to exactly reach max_duration_ms
                                   If False, drops the point that would exceed the limit (default: False)
        Returns:
            tuple:
                (clipped_x, clipped_y, clipped_t, discarded_x, discarded_y, discarded_t)
                as numpy arrays
            
        Example:
            - Input: x=[1,2,3], y=[4,5,6], t=[100, 150, 200], max_duration_ms=200
            - include_partial=False: returns x=[1], y=[4], t=[100] (only first point, total=100ms)
            - include_partial=True: returns x=[1,2], y=[4,5], t=[100, 100] (adjusted last duration to 100ms, total=200ms)
        """
        # Input validation
        assert max_duration_ms > 0, "max_duration_ms must be positive."

        # Handle empty input
        if len(traj) == 0:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
        
        traj = np.asarray(traj, dtype=np.float64)
        traj_t = traj[:, 2]
        
        # Compute cumulative time
        cumulative_time = np.cumsum(traj_t)
        
        # Find the index where cumulative time exceeds max_duration_ms
        valid_indices = np.where(cumulative_time <= max_duration_ms)[0]
        
        if len(valid_indices) == 0:
            # No points fit within the duration limit
            if include_partial and traj_t[0] > 0:
                # Include just the first point with clipped duration
                clipped = traj[:1].copy()
                clipped[0, 2] = max_duration_ms
                discarded = traj[1:].copy()
                return clipped, discarded
            else:
                return np.empty((0, 3), dtype=np.float64), traj.copy()
        
        # Get all points that fit completely
        last_valid_idx = valid_indices[-1]
        clipped = traj[:last_valid_idx + 1].copy()
        discarded = traj[last_valid_idx + 1:].copy()
        
        # Optionally include partial last point
        if include_partial and last_valid_idx < len(traj) - 1:
            remaining_time = max_duration_ms - cumulative_time[last_valid_idx]
            if remaining_time > 0:
                # Add the next point with adjusted duration
                partial_row = traj[last_valid_idx + 1].copy()
                partial_row[2] = remaining_time
                clipped = np.vstack((clipped, partial_row))
                discarded = traj[last_valid_idx + 2:].copy()
        
        return clipped, discarded

    def _convert_traj_to_fixation_count(self, traj_tp, original_width, original_height):
        """Convert TP-frame trajectory to fixation count via check_fixations."""
        traj_tp = np.asarray(traj_tp, dtype=np.float64)
        if len(traj_tp) == 0:
            return 0
        traj_xy_orig = np.column_stack(
            (
                traj_tp[:, 0] * (original_width / 512.0),
                traj_tp[:, 1] * (original_height / 320.0),
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
           to the last GT fixation (in TP coordinates).
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
    
