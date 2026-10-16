import itertools
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


def _extract_scanpath_group(
    data: Dict[Tuple[str, str], List[dict]],
    key,
    im_w: int,
    im_h: int,
):
    if isinstance(key, tuple):
        reals = data[key]
    else:
        reals = data[key]

    i_w = im_w
    i_h = im_h

    if isinstance(reals, dict) and "size" in reals and "scanpaths" in reals:
        i_w, i_h = reals["size"][1], reals["size"][0]  # type: ignore
        reals = reals["scanpaths"]  # type: ignore

    return reals, i_w, i_h


def _to_xy_array(scanpath: dict) -> np.ndarray:
    return np.array([scanpath["X"], scanpath["Y"]], dtype=np.float32).T


def _ensure_xy_array(seq) -> np.ndarray:
    arr = np.asarray(seq, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape(0, 2)
    if arr.ndim == 1:
        if arr.shape[0] == 2:
            return arr.reshape(1, 2)
        raise ValueError(f"Expected trajectory with 2 columns, got shape {arr.shape}")
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    raise ValueError(f"Expected trajectory with shape (N, 2), got shape {arr.shape}")


def _levenshtein_str_distance(a: str, b: str) -> int:
    len_a = len(a)
    len_b = len(b)
    d = [[0] * (len_b + 1) for _ in range(len_a + 1)]

    for i in range(len_a + 1):
        d[i][0] = i
    for j in range(len_b + 1):
        d[0][j] = j

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost,
            )

    return d[len_a][len_b]


def levenshtein_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    seq1 = _ensure_xy_array(seq1)
    seq2 = _ensure_xy_array(seq2)
    seq1_str = "".join([f"{int(x)}_{int(y)}," for x, y in seq1])
    seq2_str = "".join([f"{int(x)}_{int(y)}," for x, y in seq2])
    return float(_levenshtein_str_distance(seq1_str, seq2_str))


def dynamic_time_warping(seq1: np.ndarray, seq2: np.ndarray) -> float:
    seq1 = _ensure_xy_array(seq1)
    seq2 = _ensure_xy_array(seq2)

    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist = np.sqrt((seq1[i - 1, 0] - seq2[j - 1, 0]) ** 2 + (seq1[i - 1, 1] - seq2[j - 1, 1]) ** 2)
            dtw_matrix[i, j] = dist + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            )

    return float(dtw_matrix[n, m])


def discrete_frechet_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return float("nan")

    ca = np.full((n, m), -1.0, dtype=np.float64)

    def _c(i: int, j: int) -> float:
        if ca[i, j] > -1:
            return ca[i, j]

        d = float(np.linalg.norm(seq1[i] - seq2[j]))

        if i == 0 and j == 0:
            ca[i, j] = d
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(i - 1, 0), d)
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(0, j - 1), d)
        else:
            ca[i, j] = max(min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), d)

        return ca[i, j]

    return float(_c(n - 1, m - 1))


def _time_delay_embedding_distance_for_k(
    seq1: np.ndarray,
    seq2: np.ndarray,
    k: int,
    distance_mode: str = "Mean",
) -> float:
    """Compute TDE distance for a fixed embedding dimension k."""
    if len(seq1) < k or len(seq2) < k:
        return float("nan")

    seq1_vecs = [seq1[i : i + k] for i in range(len(seq1) - k + 1)]
    seq2_vecs = [seq2[i : i + k] for i in range(len(seq2) - k + 1)]

    min_distances = []
    for vec2 in seq2_vecs:
        vec2_flat = vec2.reshape(-1)
        dists = [float(np.linalg.norm(vec2_flat - vec1.reshape(-1))) for vec1 in seq1_vecs]
        if dists:
            min_distances.append(min(dists) / float(k))

    if not min_distances:
        return float("nan")

    if distance_mode == "Mean":
        return float(np.mean(min_distances))
    if distance_mode == "Hausdorff":
        return float(np.max(min_distances))
    raise ValueError(f"Unknown TDE distance_mode: {distance_mode}")


def time_delay_embedding_distance(seq1: np.ndarray, seq2: np.ndarray, im_w: int, im_h: int) -> float:
    seq1 = _ensure_xy_array(seq1)
    seq2 = _ensure_xy_array(seq2)

    if len(seq1) == 0 or len(seq2) == 0:
        return float("nan")

    # Rescale coordinates by the max stimulus dimension to make distances image-size invariant.
    max_dim = float(max(im_w, im_h))
    if max_dim <= 0:
        return float("nan")

    seq1_scaled = seq1.astype(np.float64, copy=True) / max_dim
    seq2_scaled = seq2.astype(np.float64, copy=True) / max_dim

    max_k = min(len(seq1_scaled), len(seq2_scaled))
    if max_k == 0:
        return float("nan")

    distances = []
    for k in range(1, max_k + 1):
        d = _time_delay_embedding_distance_for_k(seq1_scaled, seq2_scaled, k=k, distance_mode="Mean")
        if np.isfinite(d):
            distances.append(d)

    if not distances:
        return float("nan")

    return float(np.mean(distances))


def _pairwise_model_scores(
    human_trajs: Dict[Tuple[str, str], List[dict]],
    model_trajs: Dict[Tuple[str, str], List[dict]],
    metric_fn,
    im_w: int,
    im_h: int,
    return_scores_by_key: bool = False,
):
    all_scores: List[float] = []
    mean_scores_per_img: List[float] = []
    scores_by_key = {}

    for key in tqdm(human_trajs.keys()):
        reals, i_w, i_h = _extract_scanpath_group(human_trajs, key, im_w, im_h)
        generated, _, _ = _extract_scanpath_group(model_trajs, key, im_w, im_h)

        per_image_scores = []
        for r, g in itertools.product(reals, generated):
            gt_vector = _to_xy_array(r)
            pred_vector = _to_xy_array(g)
            score = metric_fn(gt_vector, pred_vector, i_w, i_h)
            if np.isfinite(score):
                per_image_scores.append(score)

        if per_image_scores:
            mean_scores_per_img.append(float(np.mean(per_image_scores)))
            all_scores.extend(per_image_scores)
            scores_by_key[key] = per_image_scores

    mean_score = float(np.mean(mean_scores_per_img)) if mean_scores_per_img else float("nan")
    if return_scores_by_key:
        return mean_score, all_scores, scores_by_key
    return mean_score, all_scores


def _pairwise_human_scores(
    data: Dict[Tuple[str, str], List[dict]],
    metric_fn,
    im_w: int,
    im_h: int,
):
    all_scores: List[float] = []
    mean_scores_per_img: List[float] = []

    for key in tqdm(data.keys()):
        d, i_w, i_h = _extract_scanpath_group(data, key, im_w, im_h)

        per_image_scores = []
        for r, g in itertools.combinations(d, 2):
            gt_vector = _to_xy_array(r)
            pred_vector = _to_xy_array(g)
            score = metric_fn(gt_vector, pred_vector, i_w, i_h)
            if np.isfinite(score):
                per_image_scores.append(score)

        if per_image_scores:
            mean_scores_per_img.append(float(np.mean(per_image_scores)))
            all_scores.extend(per_image_scores)

    return float(np.mean(mean_scores_per_img)) if mean_scores_per_img else float("nan"), all_scores


def _lev_wrapper(seq1: np.ndarray, seq2: np.ndarray, _im_w: int, _im_h: int) -> float:
    return levenshtein_distance(seq1, seq2)


def _dtw_wrapper(seq1: np.ndarray, seq2: np.ndarray, _im_w: int, _im_h: int) -> float:
    return dynamic_time_warping(seq1, seq2)


def _frechet_wrapper(seq1: np.ndarray, seq2: np.ndarray, _im_w: int, _im_h: int) -> float:
    return discrete_frechet_distance(seq1, seq2)


def _tde_wrapper(seq1: np.ndarray, seq2: np.ndarray, im_w: int, im_h: int) -> float:
    return time_delay_embedding_distance(seq1, seq2, im_w=im_w, im_h=im_h)


def compute_levenshtein_distance(
    human_trajs: Dict[Tuple[str, str], List[dict]],
    model_trajs: Dict[Tuple[str, str], List[dict]],
    im_w: int,
    im_h: int,
    return_scores_by_key: bool = False,
):
    return _pairwise_model_scores(
        human_trajs, model_trajs, _lev_wrapper, im_w, im_h, return_scores_by_key=return_scores_by_key
    )


def compute_self_levenshtein_distance(
    data: Dict[Tuple[str, str], List[dict]],
    im_w: int,
    im_h: int,
):
    return _pairwise_human_scores(data, _lev_wrapper, im_w, im_h)


def compute_dynamic_time_warping(
    human_trajs: Dict[Tuple[str, str], List[dict]],
    model_trajs: Dict[Tuple[str, str], List[dict]],
    im_w: int,
    im_h: int,
    return_scores_by_key: bool = False,
):
    return _pairwise_model_scores(
        human_trajs, model_trajs, _dtw_wrapper, im_w, im_h, return_scores_by_key=return_scores_by_key
    )


def compute_self_dynamic_time_warping(
    data: Dict[Tuple[str, str], List[dict]],
    im_w: int,
    im_h: int,
):
    return _pairwise_human_scores(data, _dtw_wrapper, im_w, im_h)


def compute_discrete_frechet_distance(
    human_trajs: Dict[Tuple[str, str], List[dict]],
    model_trajs: Dict[Tuple[str, str], List[dict]],
    im_w: int,
    im_h: int,
    return_scores_by_key: bool = False,
):
    return _pairwise_model_scores(
        human_trajs, model_trajs, _frechet_wrapper, im_w, im_h, return_scores_by_key=return_scores_by_key
    )


def compute_self_discrete_frechet_distance(
    data: Dict[Tuple[str, str], List[dict]],
    im_w: int,
    im_h: int,
):
    return _pairwise_human_scores(data, _frechet_wrapper, im_w, im_h)


def compute_time_delay_embedding_distance(
    human_trajs: Dict[Tuple[str, str], List[dict]],
    model_trajs: Dict[Tuple[str, str], List[dict]],
    im_w: int,
    im_h: int,
    return_scores_by_key: bool = False,
):
    return _pairwise_model_scores(
        human_trajs, model_trajs, _tde_wrapper, im_w, im_h, return_scores_by_key=return_scores_by_key
    )


def compute_self_time_delay_embedding_distance(
    data: Dict[Tuple[str, str], List[dict]],
    im_w: int,
    im_h: int,
):
    return _pairwise_human_scores(data, _tde_wrapper, im_w, im_h)
