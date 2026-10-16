from typing import Dict, Hashable, List, Sequence

import numpy as np


def similarity_to_discrepancy(scores: Sequence[float], mode: str = "one_minus") -> List[float]:
    arr = np.asarray(scores, dtype=np.float64)
    if mode == "one_minus":
        return (1.0 - arr).tolist()
    raise ValueError(f"Unknown conversion mode: {mode}")


def _normalize_scores(
    scores: Sequence[float],
    convert_higher_better: bool,
    conversion_mode: str | None,
) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr

    if convert_higher_better:
        mode = conversion_mode or "one_minus"
        arr = np.asarray(similarity_to_discrepancy(arr.tolist(), mode=mode), dtype=np.float64)
        arr = arr[np.isfinite(arr)]

    return arr


def reshape_flat_scores_by_key(
    flat_scores_by_key: Dict[Hashable, Sequence[float]],
    num_gt_by_key: Dict[Hashable, int],
    num_gen_by_key: Dict[Hashable, int],
    generated_major: bool = False,
) -> Dict[Hashable, List[List[float]]]:
    """Reshape flattened pairwise scores into per-GT score lists.

    When `generated_major=False`, flattening is assumed to be:
        for gt in GT:
            for gen in GEN:
                append(score)
    When `generated_major=True`, flattening is assumed to be:
        for gen in GEN:
            for gt in GT:
                append(score)
    """
    per_gt_scores_by_key: Dict[Hashable, List[List[float]]] = {}

    for key, flat_scores in flat_scores_by_key.items():
        h = int(num_gt_by_key.get(key, 0))
        j = int(num_gen_by_key.get(key, 0))
        if h <= 0 or j <= 0:
            continue

        flat = list(flat_scores)
        expected = h * j
        if len(flat) < expected:
            # Keep backward compatibility with existing metric code paths that may
            # skip invalid pair scores. We only use complete prefix blocks.
            expected = (len(flat) // j) * j
            if expected == 0:
                continue
            h = expected // j

        arr = np.asarray(flat[: expected], dtype=np.float64)
        if generated_major:
            matrix = arr.reshape(j, h).T  # -> (H, J)
        else:
            matrix = arr.reshape(h, j)

        per_gt_scores_by_key[key] = [row.tolist() for row in matrix]

    return per_gt_scores_by_key


def perform_crps(
    per_gt_to_gen_scores_by_key: Dict[Hashable, Sequence[Sequence[float]]],
    gen_pair_scores_by_key: Dict[Hashable, Sequence[float]],
    convert_higher_better: bool = False,
    conversion_mode: str | None = None,
) -> float:
    """Compute CRPS-like distributional metric.

    For each GT sample y:
      mean_j q(x_j, y) - (1 / (2 J^2)) * sum_{j,k} q(x_j, x_k)
    then averaged over all GT samples and keys.
    """
    per_gt_values: List[float] = []

    for key, per_gt_scores in per_gt_to_gen_scores_by_key.items():
        if not per_gt_scores:
            continue

        model_pair_scores = _normalize_scores(
            gen_pair_scores_by_key.get(key, []),
            convert_higher_better=convert_higher_better,
            conversion_mode=conversion_mode,
        )

        if model_pair_scores.size == 0:
            continue

        for gt_scores in per_gt_scores:
            gt_arr = _normalize_scores(
                gt_scores,
                convert_higher_better=convert_higher_better,
                conversion_mode=conversion_mode,
            )
            if gt_arr.size == 0:
                continue

            j = int(gt_arr.size)
            second_term = float(np.sum(model_pair_scores)) / float(2 * (j**2))
            per_gt_values.append(float(np.mean(gt_arr)) - second_term)

    if not per_gt_values:
        return float("nan")

    return float(np.mean(per_gt_values))
