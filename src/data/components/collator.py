import torch
from torch.nn.utils.rnn import pad_sequence

PAD_VALUE = -3.0


class Collator:
    """Batches samples, padding scanpaths to ``max_len_scan`` and trajectories to ``max_len_traj``."""

    def __init__(self, max_len_traj: int = 750, max_len_scan: int = 16):
        self.max_len_traj = max_len_traj
        self.max_len_scan = max_len_scan

    @staticmethod
    def _pad(sequences, max_len):
        # A dummy sequence of length max_len forces every batch to exactly max_len; it is dropped after padding.
        return pad_sequence(sequences + [torch.zeros(max_len)], batch_first=True, padding_value=PAD_VALUE)[:-1]

    def __call__(self, batch):
        imgs, tasks, task_embeddings = [], [], []
        filenames, dataset_names, original_sizes, num_viewers, question_ids = [], [], [], [], []
        scan_x, scan_y, scan_t = [], [], []
        traj_x, traj_y, traj_t, traj_fix_idx = [], [], [], []

        for s in batch:
            scanpath = s["scanpath"][: self.max_len_scan]
            scan_x.append(scanpath[:, 0])
            scan_y.append(scanpath[:, 1])
            scan_t.append(scanpath[:, 2])

            trajectory = s["trajectory"][: self.max_len_traj]
            traj_x.append(trajectory[:, 0])
            traj_y.append(trajectory[:, 1])
            traj_t.append(trajectory[:, 2])
            traj_fix_idx.append(torch.as_tensor(s["traj_fixation_index"], dtype=torch.float32)[: self.max_len_traj])

            imgs.append(s["img"].unsqueeze(0))
            tasks.append(s["task"])
            task_embeddings.append(s["task_embedding"].unsqueeze(0))
            filenames.append(s["img_filename"])
            dataset_names.append(s["dataset"])
            original_sizes.append(s["original_img_size"])
            num_viewers.append(s["num_viewers"])
            question_ids.append(s["question_id"])

        scanpath = torch.stack(
            [self._pad(scan_x, self.max_len_scan), self._pad(scan_y, self.max_len_scan), self._pad(scan_t, self.max_len_scan)],
            dim=2,
        )
        trajectory = torch.stack(
            [self._pad(traj_x, self.max_len_traj), self._pad(traj_y, self.max_len_traj), self._pad(traj_t, self.max_len_traj)],
            dim=2,
        )
        return {
            "img_filename": filenames,
            "original_img_size": original_sizes,
            "img": torch.cat(imgs, dim=0),
            "task": tasks,
            "task_embedding": torch.cat(task_embeddings, dim=0),
            "num_viewers": num_viewers,
            "dataset": dataset_names,
            "question_id": question_ids,
            "scanpath": scanpath,
            "padding_mask": torch.where(scanpath[:, :, 0] == PAD_VALUE, 0, 1),
            "trajectory": trajectory,
            "trajectory_padding_mask": torch.where(trajectory[:, :, 0] == PAD_VALUE, 0, 1),
            "traj_fixation_index": self._pad(traj_fix_idx, self.max_len_traj),
        }
