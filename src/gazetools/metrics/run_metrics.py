import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm

from src.gazetools.metrics.sequence_score import (
    compute_SS, compute_SSS, 
    compute_self_SS, compute_self_SSS,
    compute_SS_Time, compute_DSS_Time, compute_DSS_no_dur, compute_self_SS_Time,
    compute_SSS_Time, compute_self_SSS_Time,
    compute_SS_model_vs_model, compute_SS_Time_model_vs_model,
)

from src.gazetools.density import (
    compute_lengths_density_chart,
    compute_durations_density_chart,
    compute_marks_density_chart,
    compute_saccades_amplitude_chart,
    compute_saccades_direction_chart,
)
from src.gazetools.metrics.kld import perform_kld
from src.gazetools.metrics.crps import perform_crps, reshape_flat_scores_by_key
from src.gazetools.metrics.multimatch import (
    compute_multi_match_score,
    compute_self_multi_match_score,
)
from src.gazetools.metrics.saliency import (
    compute_saliency_metrics,
    compute_self_saliency_metrics,
)

from src.gazetools.metrics.scanmatch import scan_match_score, self_scan_match_score
from src.gazetools.metrics.string_edit_distance import compute_string_edit_distance, compute_self_string_edit_distance
from src.gazetools.metrics.trajectory_distances import (
    compute_levenshtein_distance,
    compute_self_levenshtein_distance,
    compute_dynamic_time_warping,
    compute_self_dynamic_time_warping,
    compute_discrete_frechet_distance,
    compute_self_discrete_frechet_distance,
    compute_time_delay_embedding_distance,
    compute_self_time_delay_embedding_distance,
)


def _safe_best(scores: List[float], higher_is_better: bool) -> float:
    if not scores:
        return float("nan")
    return float(np.max(scores) if higher_is_better else np.min(scores))


def _mean_of_per_key_bests(scores_by_key: dict, higher_is_better: bool) -> float:
    per_key_bests = []
    for _, scores in scores_by_key.items():
        valid_scores = [s for s in scores if np.isfinite(s)]
        if not valid_scores:
            continue
        per_key_bests.append(max(valid_scores) if higher_is_better else min(valid_scores))
    if not per_key_bests:
        return float("nan")
    return float(np.mean(per_key_bests))


def _mean_of_per_gt_bests(
    scores_by_key: dict,
    num_gt_by_key: dict,
    num_gen_by_key: dict,
    higher_is_better: bool,
    generated_major: bool = False,
) -> float:
    """For each GT scanpath, take best score across generated samples, then average over GTs."""
    try:
        per_gt_scores_by_key = reshape_flat_scores_by_key(
            flat_scores_by_key=scores_by_key,
            num_gt_by_key=num_gt_by_key,
            num_gen_by_key=num_gen_by_key,
            generated_major=generated_major,
        )
    except Exception:
        # Fallback for metrics whose score layout is not flat pairwise GTxGEN.
        return _mean_of_per_key_bests(scores_by_key, higher_is_better)

    per_gt_bests = []
    for _, per_gt_scores in per_gt_scores_by_key.items():
        for gt_scores in per_gt_scores:
            valid_scores = [s for s in gt_scores if np.isfinite(s)]
            if not valid_scores:
                continue
            per_gt_bests.append(max(valid_scores) if higher_is_better else min(valid_scores))

    if not per_gt_bests:
        return float("nan")
    return float(np.mean(per_gt_bests))


def _num_scanpaths_by_key(data: dict) -> dict:
    counts = {}
    for key, value in data.items():
        if isinstance(value, dict) and "scanpaths" in value:
            counts[key] = len(value["scanpaths"])
        else:
            counts[key] = len(value)
    return counts


def _sequence_condition_key(scanpath: dict, is_validation: bool) -> str:
    split = "validation" if is_validation else "test"
    dataset = scanpath["dataset"]
    if dataset in ["cocosearch18_tp", "cocosearch18_ta", "cocosearch18_edf_tp", "cocosearch18_edf_ta"]:
        task = scanpath["task"].replace("_", " ")
        return f"{split}-{scanpath['condition']}-{task}-{scanpath['name']}"
    if dataset == "aird":
        return f"{split}-{scanpath['question_id']}-{scanpath['name']}"
    if dataset in ["osie", "cocofreeview", "cocofreeview_edf", "mit1003", "mit1003_edf"]:
        return f"{split}-{scanpath['name']}"
    raise NotImplementedError("Dataset not implemented")


def _expand_symmetric_combinations_to_ordered_with_self(
    pair_scores_by_key: dict,
    num_gen_by_key: dict,
    self_score: float,
) -> dict:
    expanded = {}
    for key, scores in pair_scores_by_key.items():
        j = int(num_gen_by_key.get(key, 0))
        if j <= 0:
            continue
        values = [self_score for _ in range(j)]
        for score in scores:
            values.append(score)
            values.append(score)
        expanded[key] = values
    return expanded

def run_density(
    originals: Dict[str, List[dict]],
    generated: Dict[str, List[dict]],
    save_directory: str,
):

    originals_list = list(
        itertools.chain.from_iterable(map(lambda x: x[1], originals.items()))
    )
    generated_list = list(
        itertools.chain.from_iterable(map(lambda x: x[1], generated.items()))
    )

    scanpath_lengths_original = [len(originals_list[i]['X']) for i in range(len(originals_list))]
    scanpath_lengths_generated = [len(generated_list[i]['X']) for i in range(len(generated_list))]
    
    original_xs: List[float] = list(
        itertools.chain.from_iterable(
            map(
                lambda x: itertools.chain.from_iterable((map(lambda x: x["X"], x[1]))),
                originals.items(),
            )
        )
    )
    original_ys: List[float] = list(
        itertools.chain.from_iterable(
            map(
                lambda x: itertools.chain.from_iterable((map(lambda x: x["Y"], x[1]))),
                originals.items(),
            )
        )
    )
    original_ts: List[float] = list(
        itertools.chain.from_iterable(
            map(
                lambda x: itertools.chain.from_iterable((map(lambda x: x["T"], x[1]))),
                originals.items(),
            )
        )
    )

    generated_xs: List[float] = list(
        itertools.chain.from_iterable(
            map(
                lambda x: itertools.chain.from_iterable((map(lambda x: x["X"], x[1]))),
                generated.items(),
            )
        )
    )
    generated_ys: List[float] = list(
        itertools.chain.from_iterable(
            map(
                lambda x: itertools.chain.from_iterable((map(lambda x: x["Y"], x[1]))),
                generated.items(),
            )
        )
    )
    generated_ts: List[float] = list(
        itertools.chain.from_iterable(
            map(
                lambda x: itertools.chain.from_iterable((map(lambda x: x["T"], x[1]))),
                generated.items(),
            )
        )
    )

    scanpath_lengths_original = [len(originals_list[i]['X']) for i in range(len(originals_list))]
    scanpath_lengths_generated = [len(generated_list[i]['X']) for i in range(len(generated_list))]
        
    print("Generating density charts...")
    compute_lengths_density_chart(scanpath_lengths_original, scanpath_lengths_generated, os.path.join(save_directory, "density_scanpath_lengths.png"))
    
    compute_durations_density_chart(
        original_ts,
        generated_ts,
        os.path.join(save_directory, "density_durations.png"),
    )

    compute_marks_density_chart(
        (original_xs, original_ys),
        (generated_xs, generated_ys),
        os.path.join(save_directory, "density_marks.png"),
    )

    o = list(map(lambda x: (x["X"], x["Y"]), originals_list))
    g = list(map(lambda x: (x["X"], x["Y"]), generated_list))

    compute_saccades_amplitude_chart(
        o,
        g,
        os.path.join(save_directory, "density_amplitude.png"),
    )

    compute_saccades_direction_chart(
        o, g, os.path.join(save_directory, "density_direction.png")
    )

    print("Done.")


def run_metrics(
    originals: dict,
    generated: dict,
    datamodule = None,
    metrics_list: List[str] = [],
    ss_clusters = None,
    data_root = None,
    images_width=1680,
    images_height=1050,
    is_validation=False
):
    #NB: this mean values is not the correct way to compute...
    #it should compute the mean across subject of an image. Then another mean across all images. 
    #Instead of doing a single global mean. However results should not be very different. 
    #Instead the KLD i still correct. 
    metrics = pd.DataFrame()
    num_gt_by_key = _num_scanpaths_by_key(originals)
    num_gen_by_key = _num_scanpaths_by_key(generated)
    dataset_names = {
        value.get("dataset")
        for value in originals.values()
        if isinstance(value, dict) and value.get("dataset") is not None
    }
    freeview_datasets = {"osie", "cocofreeview", "cocofreeview_edf", "mit1003", "mit1003_edf"}

    if "multi_match" in metrics_list or len(metrics_list) == 0:
        print("Generating Model Multi Match metrics...")
        generated_mm, generated_mm_by_key = compute_multi_match_score(
           originals, generated, images_width, images_height, return_scores_by_key=True
        )
        print("Done.")

        print("Generating Human Multi Match metrics...")
        human_mm = compute_self_multi_match_score(
            originals, images_width, images_height
        )
        print("Done.")

        g_vectors = [x[0] for x in generated_mm]
        g_directions = [x[1] for x in generated_mm]
        g_lengths = [x[2] for x in generated_mm]
        g_positions = [x[3] for x in generated_mm]
        g_durations = [x[4] for x in generated_mm]
        g_mms = [np.mean(x) for x in generated_mm]

        g_vectors_by_key = {
            key: [float(score[0]) for score in scores]
            for key, scores in generated_mm_by_key.items()
        }
        g_directions_by_key = {
            key: [float(score[1]) for score in scores]
            for key, scores in generated_mm_by_key.items()
        }
        g_lengths_by_key = {
            key: [float(score[2]) for score in scores]
            for key, scores in generated_mm_by_key.items()
        }
        g_positions_by_key = {
            key: [float(score[3]) for score in scores]
            for key, scores in generated_mm_by_key.items()
        }
        g_durations_by_key = {
            key: [float(score[4]) for score in scores]
            for key, scores in generated_mm_by_key.items()
        }
        g_mms_by_key = {
            key: [float(np.mean(score)) for score in scores]
            for key, scores in generated_mm_by_key.items()
        }

        g_vector = np.mean(g_vectors)
        g_direction = np.mean(g_directions)
        g_length = np.mean(g_lengths)
        g_position = np.mean(g_positions)
        g_duration = np.mean(g_durations)
        g_mm = np.mean(g_mms)

        metrics.loc["Model", "MM_Vector"] = g_vector  # type: ignore
        metrics.loc["Model", "MM_VectorBest"] = _mean_of_per_gt_bests(g_vectors_by_key, num_gt_by_key, num_gen_by_key, higher_is_better=True)  # type: ignore
        metrics.loc["Model", "MM_Direction"] = g_direction  # type: ignore
        metrics.loc["Model", "MM_DirectionBest"] = _mean_of_per_gt_bests(g_directions_by_key, num_gt_by_key, num_gen_by_key, higher_is_better=True)  # type: ignore
        metrics.loc["Model", "MM_Length"] = g_length  # type: ignore
        metrics.loc["Model", "MM_LengthBest"] = _mean_of_per_gt_bests(g_lengths_by_key, num_gt_by_key, num_gen_by_key, higher_is_better=True)  # type: ignore
        metrics.loc["Model", "MM_Position"] = g_position  # type: ignore
        metrics.loc["Model", "MM_PositionBest"] = _mean_of_per_gt_bests(g_positions_by_key, num_gt_by_key, num_gen_by_key, higher_is_better=True)  # type: ignore
        metrics.loc["Model", "MM_Duration"] = g_duration  # type: ignore
        metrics.loc["Model", "MM_DurationBest"] = _mean_of_per_gt_bests(g_durations_by_key, num_gt_by_key, num_gen_by_key, higher_is_better=True)  # type: ignore
        metrics.loc["Model", "MultiMatch"] = g_mm  # type: ignore
        metrics.loc["Model", "MultiMatchBest"] = _mean_of_per_gt_bests(g_mms_by_key, num_gt_by_key, num_gen_by_key, higher_is_better=True)  # type: ignore

        h_vectors = [x[0] for x in human_mm]
        h_directions = [x[1] for x in human_mm]
        h_lengths = [x[2] for x in human_mm]
        h_positions = [x[3] for x in human_mm]
        h_durations = [x[4] for x in human_mm]
        h_mms = [np.mean(x) for x in human_mm]

        h_vector = np.mean(h_vectors)
        h_direction = np.mean(h_directions)
        h_length = np.mean(h_lengths)
        h_position = np.mean(h_positions)
        h_duration = np.mean(h_durations)
        h_mm = np.mean(h_mms)

        metrics.loc[f"Human", "MM_Vector"] = h_vector  # type: ignore
        metrics.loc[f"Human", "MM_Direction"] = h_direction  # type: ignore
        metrics.loc[f"Human", "MM_Length"] = h_length  # type: ignore
        metrics.loc[f"Human", "MM_Position"] = h_position  # type: ignore
        metrics.loc[f"Human", "MM_Duration"] = h_duration  # type: ignore
        metrics.loc[f"Human", "MultiMatch"] = h_mm  # type: ignore

        if "kld" in metrics_list or len(metrics_list) == 0:
            print("Calculating KDL in Multi Match metrics...")
            
            k_vector = perform_kld(h_vectors, g_vectors)
            k_direction = perform_kld(h_directions, g_directions)
            k_length = perform_kld(h_lengths, g_lengths)
            k_position = perform_kld(h_positions, g_positions)
            k_duration = perform_kld(h_durations, g_durations)
            k_mm = np.mean([k_vector, k_direction, k_length, k_position, k_duration])

            metrics.loc["KLD", "MM_Vector"] = k_vector
            metrics.loc["KLD", "MM_Direction"] = k_direction
            metrics.loc["KLD", "MM_Length"] = k_length
            metrics.loc["KLD", "MM_Position"] = k_position
            metrics.loc["KLD", "MM_Duration"] = k_duration
            metrics.loc["KLD", "MultiMatch"] = k_mm  # type: ignore

        if "crps" in metrics_list or len(metrics_list) == 0:
            print("Calculating CRPS in Multi Match metrics...")
            generated_generated_mm, generated_generated_mm_by_key = compute_multi_match_score(
                generated, generated, images_width, images_height, return_scores_by_key=True
            )
            _ = generated_generated_mm  # keep signature explicit
            num_gt_by_key = _num_scanpaths_by_key(originals)
            num_gen_by_key = _num_scanpaths_by_key(generated)

            component_names = [
                "MM_Vector",
                "MM_Direction",
                "MM_Length",
                "MM_Position",
                "MM_Duration",
            ]
            component_indices = [0, 1, 2, 3, 4]
            component_crps = []

            for metric_name, idx in zip(component_names, component_indices):
                flat_gt_gen_by_key = {
                    key: [float(score[idx]) for score in scores]
                    for key, scores in generated_mm_by_key.items()
                }
                flat_gen_gen_by_key = {
                    key: [float(score[idx]) for score in scores]
                    for key, scores in generated_generated_mm_by_key.items()
                }
                per_gt_scores_by_key = reshape_flat_scores_by_key(
                    flat_scores_by_key=flat_gt_gen_by_key,
                    num_gt_by_key=num_gt_by_key,
                    num_gen_by_key=num_gen_by_key,
                    generated_major=False,
                )
                c_val = perform_crps(
                    per_gt_to_gen_scores_by_key=per_gt_scores_by_key,
                    gen_pair_scores_by_key=flat_gen_gen_by_key,
                    convert_higher_better=False,
                    conversion_mode=None,
                )
                metrics.loc["CRPS", metric_name] = c_val
                component_crps.append(c_val)

            metrics.loc["CRPS", "MultiMatch"] = float(np.nanmean(component_crps))  # type: ignore

    if "scan_match" in metrics_list or len(metrics_list) == 0:
        print("Generating Model Scan Match No Duration metrics...")
        g_scores, g_scores_by_key = scan_match_score(
            real=originals,
            generated=generated,
            stimulus_width=images_width,
            stimulus_height=images_height,
            tempbin=0,
            return_scores_by_key=True,
        )
        metrics.loc["Model", "ScanMatch"] = np.mean(g_scores)  # type: ignore
        metrics.loc["Model", "ScanMatchBest"] = _mean_of_per_gt_bests(g_scores_by_key, num_gt_by_key, num_gen_by_key, higher_is_better=True)  # type: ignore
        print("Done.")

        print("Generating Human Scan Match No Duration metrics...")
        h_scores = self_scan_match_score(
            data=originals,
            stimulus_width=images_width,
            stimulus_height=images_height,
            tempbin=0,
        )
        metrics.loc[f"Human", "ScanMatch"] = np.mean(h_scores)  # type: ignore
        print("Done.")

        if "kld" in metrics_list or len(metrics_list) == 0:
            print("Calculating KDL in ScanMatch No Duration...")
            
            metrics.loc["KLD", "ScanMatch"] = perform_kld(h_scores, g_scores)

        if "crps" in metrics_list or len(metrics_list) == 0:
            print("Calculating CRPS in ScanMatch No Duration...")
            _, g2g_scores_by_key = scan_match_score(
                real=generated,
                generated=generated,
                stimulus_width=images_width,
                stimulus_height=images_height,
                tempbin=0,
                return_scores_by_key=True,
            )
            per_gt_scores_by_key = reshape_flat_scores_by_key(
                flat_scores_by_key=g_scores_by_key,
                num_gt_by_key=_num_scanpaths_by_key(originals),
                num_gen_by_key=_num_scanpaths_by_key(generated),
                generated_major=False,
            )
            metrics.loc["CRPS", "ScanMatch"] = perform_crps(
                per_gt_to_gen_scores_by_key=per_gt_scores_by_key,
                gen_pair_scores_by_key=g2g_scores_by_key,
                convert_higher_better=False,
                conversion_mode=None,
            )

    if "scan_match" in metrics_list or len(metrics_list) == 0:
        print("Generating Model Scan Match metric...")
        g_scores, g_scores_by_key = scan_match_score(
           real=originals,
           generated=generated,
           stimulus_width=images_width,
           stimulus_height=images_height,
           tempbin=50,
           return_scores_by_key=True,
        )
        metrics.loc["Model", "ScanMatchTime"] = np.mean(g_scores)  # type: ignore
        metrics.loc["Model", "ScanMatchTimeBest"] = _mean_of_per_gt_bests(g_scores_by_key, num_gt_by_key, num_gen_by_key, higher_is_better=True)  # type: ignore
        print("Done.")

        print("Generating Human Scan Match metric...")
        h_scores = self_scan_match_score(
            data=originals,
            stimulus_width=images_width,
            stimulus_height=images_height,
            tempbin=50,
        )
        metrics.loc[f"Human", "ScanMatchTime"] = np.mean(h_scores)  # type: ignore
        print("Done.")

        if "kld" in metrics_list or len(metrics_list) == 0:
            print("Calculating KDL in ScanMatch...")
            metrics.loc["KLD", "ScanMatchTime"] = perform_kld(h_scores, g_scores)

        if "crps" in metrics_list or len(metrics_list) == 0:
            print("Calculating CRPS in ScanMatch Time...")
            _, g2g_scores_by_key = scan_match_score(
                real=generated,
                generated=generated,
                stimulus_width=images_width,
                stimulus_height=images_height,
                tempbin=50,
                return_scores_by_key=True,
            )
            per_gt_scores_by_key = reshape_flat_scores_by_key(
                flat_scores_by_key=g_scores_by_key,
                num_gt_by_key=_num_scanpaths_by_key(originals),
                num_gen_by_key=_num_scanpaths_by_key(generated),
                generated_major=False,
            )
            metrics.loc["CRPS", "ScanMatchTime"] = perform_crps(
                per_gt_to_gen_scores_by_key=per_gt_scores_by_key,
                gen_pair_scores_by_key=g2g_scores_by_key,
                convert_higher_better=False,
                conversion_mode=None,
            )

    if "sed" in metrics_list or len(metrics_list) == 0:
        print('Generating Model String Edit Distance metrics...')
        sed = compute_string_edit_distance(originals, generated, images_width, images_height)
        metrics.loc["Model", "SED"] = sed  # type: ignore
        
        #print('Generating Human String Edit Distance metrics...')
        #h_scores = compute_self_string_edit_distance(originals, images_width, images_height)
        #metrics.loc["Human", "SED"] = np.mean(h_scores)  # type: ignore
        
        #if "kld" in metrics_list or len(metrics_list) == 0:
            #print("Calculating KDL in SDE...")
            #metrics.loc["KLD", "SED"] = perform_kld(h_scores, g_scores)
        print("Done.")

    if (
        "levenshtein_distance" in metrics_list
        or len(metrics_list) == 0
    ):
        print('Generating Model Levenshtein Distance metrics...')
        g_score, g_scores, g_scores_by_key = compute_levenshtein_distance(
            originals, generated, images_width, images_height, return_scores_by_key=True
        )
        metrics.loc["Model", "LevenshteinDistance"] = g_score  # type: ignore
        metrics.loc["Model", "LevenshteinDistanceBest"] = _mean_of_per_gt_bests(g_scores_by_key, num_gt_by_key, num_gen_by_key, higher_is_better=False)  # type: ignore

        print('Generating Human Levenshtein Distance metrics...')
        h_score, h_scores = compute_self_levenshtein_distance(
            originals, images_width, images_height
        )
        metrics.loc["Human", "LevenshteinDistance"] = h_score  # type: ignore
        metrics.loc["Human", "LevenshteinDistanceBest"] = _safe_best(h_scores, higher_is_better=False)  # type: ignore

        if "kld" in metrics_list or len(metrics_list) == 0:
            print("Calculating KLD in Levenshtein Distance...")
            metrics.loc["KLD", "LevenshteinDistance"] = perform_kld(h_scores, g_scores)

        if "crps" in metrics_list or len(metrics_list) == 0:
            print("Calculating CRPS in Levenshtein Distance...")
            _, _, g2g_scores_by_key = compute_levenshtein_distance(
                generated, generated, images_width, images_height, return_scores_by_key=True
            )
            per_gt_scores_by_key = reshape_flat_scores_by_key(
                flat_scores_by_key=g_scores_by_key,
                num_gt_by_key=_num_scanpaths_by_key(originals),
                num_gen_by_key=_num_scanpaths_by_key(generated),
                generated_major=False,
            )
            metrics.loc["CRPS", "LevenshteinDistance"] = perform_crps(
                per_gt_to_gen_scores_by_key=per_gt_scores_by_key,
                gen_pair_scores_by_key=g2g_scores_by_key,
                convert_higher_better=False,
                conversion_mode=None,
            )

    if "dynamic_time_warping" in metrics_list or len(metrics_list) == 0:
        print('Generating Model Dynamic Time Warping metrics...')
        g_score, g_scores, g_scores_by_key = compute_dynamic_time_warping(
            originals, generated, images_width, images_height, return_scores_by_key=True
        )
        metrics.loc["Model", "DynamicTimeWarping"] = g_score  # type: ignore
        metrics.loc["Model", "DynamicTimeWarpingBest"] = _mean_of_per_gt_bests(g_scores_by_key, num_gt_by_key, num_gen_by_key, higher_is_better=False)  # type: ignore

        print('Generating Human Dynamic Time Warping metrics...')
        h_score, h_scores = compute_self_dynamic_time_warping(
            originals, images_width, images_height
        )
        metrics.loc["Human", "DynamicTimeWarping"] = h_score  # type: ignore
        metrics.loc["Human", "DynamicTimeWarpingBest"] = _safe_best(h_scores, higher_is_better=False)  # type: ignore

        if "kld" in metrics_list or len(metrics_list) == 0:
            print("Calculating KLD in Dynamic Time Warping...")
            metrics.loc["KLD", "DynamicTimeWarping"] = perform_kld(h_scores, g_scores)

        if "crps" in metrics_list or len(metrics_list) == 0:
            print("Calculating CRPS in Dynamic Time Warping...")
            _, _, g2g_scores_by_key = compute_dynamic_time_warping(
                generated, generated, images_width, images_height, return_scores_by_key=True
            )
            per_gt_scores_by_key = reshape_flat_scores_by_key(
                flat_scores_by_key=g_scores_by_key,
                num_gt_by_key=_num_scanpaths_by_key(originals),
                num_gen_by_key=_num_scanpaths_by_key(generated),
                generated_major=False,
            )
            metrics.loc["CRPS", "DynamicTimeWarping"] = perform_crps(
                per_gt_to_gen_scores_by_key=per_gt_scores_by_key,
                gen_pair_scores_by_key=g2g_scores_by_key,
                convert_higher_better=False,
                conversion_mode=None,
            )

    if "discrete_frechet_distance" in metrics_list or len(metrics_list) == 0:
        print('Generating Model Discrete Frechet Distance metrics...')
        g_score, g_scores, g_scores_by_key = compute_discrete_frechet_distance(
            originals, generated, images_width, images_height, return_scores_by_key=True
        )
        metrics.loc["Model", "DiscreteFrechetDistance"] = g_score  # type: ignore
        metrics.loc["Model", "DiscreteFrechetDistanceBest"] = _mean_of_per_gt_bests(g_scores_by_key, num_gt_by_key, num_gen_by_key, higher_is_better=False)  # type: ignore

        print('Generating Human Discrete Frechet Distance metrics...')
        h_score, h_scores = compute_self_discrete_frechet_distance(
            originals, images_width, images_height
        )
        metrics.loc["Human", "DiscreteFrechetDistance"] = h_score  # type: ignore
        metrics.loc["Human", "DiscreteFrechetDistanceBest"] = _safe_best(h_scores, higher_is_better=False)  # type: ignore

        if "kld" in metrics_list or len(metrics_list) == 0:
            print("Calculating KLD in Discrete Frechet Distance...")
            metrics.loc["KLD", "DiscreteFrechetDistance"] = perform_kld(h_scores, g_scores)

        if "crps" in metrics_list or len(metrics_list) == 0:
            print("Calculating CRPS in Discrete Frechet Distance...")
            _, _, g2g_scores_by_key = compute_discrete_frechet_distance(
                generated, generated, images_width, images_height, return_scores_by_key=True
            )
            per_gt_scores_by_key = reshape_flat_scores_by_key(
                flat_scores_by_key=g_scores_by_key,
                num_gt_by_key=_num_scanpaths_by_key(originals),
                num_gen_by_key=_num_scanpaths_by_key(generated),
                generated_major=False,
            )
            metrics.loc["CRPS", "DiscreteFrechetDistance"] = perform_crps(
                per_gt_to_gen_scores_by_key=per_gt_scores_by_key,
                gen_pair_scores_by_key=g2g_scores_by_key,
                convert_higher_better=False,
                conversion_mode=None,
            )

    if "time_delay_embedding" in metrics_list or len(metrics_list) == 0:
        print('Generating Model Time Delay Embedding metrics...')
        g_score, g_scores, g_scores_by_key = compute_time_delay_embedding_distance(
            originals, generated, images_width, images_height, return_scores_by_key=True
        )
        metrics.loc["Model", "TimeDelayEmbedding"] = g_score  # type: ignore
        metrics.loc["Model", "TimeDelayEmbeddingBest"] = _mean_of_per_gt_bests(g_scores_by_key, num_gt_by_key, num_gen_by_key, higher_is_better=False)  # type: ignore

        print('Generating Human Time Delay Embedding metrics...')
        h_score, h_scores = compute_self_time_delay_embedding_distance(
            originals, images_width, images_height
        )
        metrics.loc["Human", "TimeDelayEmbedding"] = h_score  # type: ignore
        metrics.loc["Human", "TimeDelayEmbeddingBest"] = _safe_best(h_scores, higher_is_better=False)  # type: ignore

        if "kld" in metrics_list or len(metrics_list) == 0:
            print("Calculating KLD in Time Delay Embedding...")
            metrics.loc["KLD", "TimeDelayEmbedding"] = perform_kld(h_scores, g_scores)

        if "crps" in metrics_list or len(metrics_list) == 0:
            print("Calculating CRPS in Time Delay Embedding...")
            _, _, g2g_scores_by_key = compute_time_delay_embedding_distance(
                generated, generated, images_width, images_height, return_scores_by_key=True
            )
            per_gt_scores_by_key = reshape_flat_scores_by_key(
                flat_scores_by_key=g_scores_by_key,
                num_gt_by_key=_num_scanpaths_by_key(originals),
                num_gen_by_key=_num_scanpaths_by_key(generated),
                generated_major=False,
            )
            metrics.loc["CRPS", "TimeDelayEmbedding"] = perform_crps(
                per_gt_to_gen_scores_by_key=per_gt_scores_by_key,
                gen_pair_scores_by_key=g2g_scores_by_key,
                convert_higher_better=False,
                conversion_mode=None,
            )
    
    DSS_per_img_no_dur = None
    covered_human_scanpaths_no_dur = None
    if "sequence_score" in metrics_list or len(metrics_list) == 0:
        #compute sequence score
        print('Generating Model Sequence Score metrics...')
        # adjust predictions for SS computation
        modified_generations = []
        for key, value in tqdm(generated.items()):
            for sample in value['scanpaths']:
                scanpath = {}
                
                if isinstance(key, tuple):
                    scanpath['name'] = str(Path(key[0]).stem)
                    scanpath['task'] = key[1]
                else:
                    scanpath['name'] = str(Path(key).stem)
                
                scanpath['dataset'] = value['dataset']
                
                if scanpath['dataset'] in ['cocosearch18_tp', 'cocosearch18_edf_tp']:
                    scanpath['condition'] = 'present'
                elif scanpath['dataset'] in ['cocosearch18_ta', 'cocosearch18_edf_ta']:
                    scanpath['condition'] = 'absent'
                else:
                    scanpath['condition'] = ''
                
                scanpath['question_id'] = value['question_id']

                scanpath['X'] = sample['X']
                scanpath['Y'] = sample['Y']
                if 'T' in sample:
                    scanpath['T'] = sample['T']
                modified_generations.append(scanpath)
        
        if "sequence_score" in metrics_list or len(metrics_list) == 0:
            threshold_no_dur = datamodule.test_datasets[0].threshold_no_dur
            g_sequence_score_results, g_scores, scanpath_recall, covered_human_scanpaths_no_dur, g_scores_by_key = compute_SS(
                modified_generations,
                ss_clusters,
                truncate=16,
                is_validation=is_validation,
                threshold_no_dur=threshold_no_dur,
                return_scores_by_key=True,
            )
            ss_final_score = np.mean([r['score'] for r in g_sequence_score_results])
            metrics.loc["Model", "SequenceScore"] = ss_final_score
            metrics.loc["Model", "SequenceScoreBest"] = _mean_of_per_gt_bests(g_scores_by_key, num_gt_by_key, num_gen_by_key, higher_is_better=True)
            metrics.loc["Scanpath_Recall", "SequenceScore"] = scanpath_recall
            
            h_sequence_score_results, h_scores = compute_self_SS(originals, ss_clusters, truncate=16, is_validation=is_validation)
            ss_final_score = np.mean([r['score'] for r in h_sequence_score_results])
            metrics.loc["Human", "SequenceScore"] = ss_final_score
        
        if "kld" in metrics_list or len(metrics_list) == 0:
            print("Calculating KDL in Sequence Score...")
            metrics.loc["KLD", "SequenceScore"] = perform_kld(h_scores, g_scores)

        if "crps" in metrics_list or len(metrics_list) == 0:
            print("Calculating CRPS in Sequence Score...")
            generated_by_condition_key = {}
            for scanpath in modified_generations:
                condition_key = _sequence_condition_key(scanpath, is_validation=is_validation)
                generated_by_condition_key.setdefault(condition_key, []).append(scanpath)

            num_gen_by_key = {
                key: len(scanpaths)
                for key, scanpaths in generated_by_condition_key.items()
            }
            num_gt_by_key = {
                key: len(ss_clusters[key]["strings"])
                for key in g_scores_by_key.keys()
                if key in ss_clusters
            }

            g2g_scores_by_key = {}
            for key, scanpaths in generated_by_condition_key.items():
                if len(scanpaths) < 2:
                    continue
                g2g_scores_by_key[key] = compute_SS_model_vs_model(
                    scanpaths,
                    clusters=ss_clusters,
                    threshold_no_dur=threshold_no_dur,
                    truncate=16,
                    is_validation=is_validation,
                )

            g2g_ordered_by_key = _expand_symmetric_combinations_to_ordered_with_self(
                pair_scores_by_key=g2g_scores_by_key,
                num_gen_by_key=num_gen_by_key,
                self_score=1.0,
            )
            per_gt_scores_by_key = reshape_flat_scores_by_key(
                flat_scores_by_key=g_scores_by_key,
                num_gt_by_key=num_gt_by_key,
                num_gen_by_key=num_gen_by_key,
                generated_major=True,
            )
            metrics.loc["CRPS", "SequenceScore"] = perform_crps(
                per_gt_to_gen_scores_by_key=per_gt_scores_by_key,
                gen_pair_scores_by_key=g2g_ordered_by_key,
                convert_higher_better=False,
                conversion_mode=None,
            )
        
        if "diversity_sequence_score" in metrics_list or len(metrics_list) == 0:
            print('Generating Model Diversity Sequence Score metrics...')
            threshold = datamodule.test_datasets[0].threshold
            DSS_no_dur, DSS_per_img_no_dur = compute_DSS_no_dur(modified_generations, clusters=ss_clusters, threshold_no_dur=threshold_no_dur,
                                                                truncate=16, is_validation=is_validation)
            metrics.loc["Model", "DSS"] = DSS_no_dur
        
        print('Done!')
        
        
    all_human_vs_human, all_human_vs_model, all_model_vs_model = None, None, None
    DSS_per_img = None
    covered_human_scanpaths = None
    if (
        "sequence_score" in metrics_list
        or "diversity_sequence_score" in metrics_list
        or len(metrics_list) == 0
    ):
        #compute sequence score
        print('Generating Model Sequence Score Time metrics...')
        #adjust predictions for SS computation
        modified_generations = []
        for key, value in tqdm(generated.items()):
            for sample in value['scanpaths']:
                scanpath = {}
                
                if isinstance(key, tuple):
                    scanpath['name'] = str(Path(key[0]).stem)
                    scanpath['task'] = key[1]
                else:
                    scanpath['name'] = str(Path(key).stem)    
            
                scanpath['dataset'] = value['dataset']
                
                if scanpath['dataset'] in ['cocosearch18_tp', 'cocosearch18_edf_tp']:
                    scanpath['condition'] = 'present'
                elif scanpath['dataset'] in ['cocosearch18_ta', 'cocosearch18_edf_ta']:
                    scanpath['condition'] = 'absent'
                else:
                    scanpath['condition'] = ''
                
                scanpath['question_id'] = value['question_id']

                scanpath['X'] = sample['X']
                scanpath['Y'] = sample['Y']
                scanpath['T'] = sample['T']
                modified_generations.append(scanpath)
                
        t_dict = {}
        split = 'validation' if is_validation else 'test'
        
        for traj in ss_clusters:
            if split not in traj:
                continue
            if len(traj.split('-')) == 4: # visual search case
                task = traj.split('-')[2]
            for idx, (subj, sample) in enumerate(ss_clusters[traj]['strings'].items()):
                name = traj.split('-')[-1]
                if len(traj.split('-')) == 4 and task != '': # visual search case
                    condition = traj.split('-')[1] # absent or present
                    key = '{}-{}-{}-{}-{}'.format(split, condition, task, name, subj) # subject count starts from 1 instead of 0 in the clusters
                else: #freeviewing case, both with and without empty string as embedding
                    if isinstance(traj, tuple): #freeview with task embedding
                        key = '{}-{}-{}'.format(split, str(Path(traj[0]).stem), subj)
                    else:
                        if len(traj.split('-')) == 3: # aird case
                            question_id = traj.split('-')[1]
                            img_id = traj.split('-')[2]
                            key = '{}-{}-{}-{}'.format(split, question_id, img_id, subj)
                        else:
                            key = '{}-{}-{}'.format(split, name, subj) # subject count starts from 1 instead of 0 in the clusters
                
                times = ss_clusters[traj]['times'][idx]
                t_dict[key] = np.array(times)        
        if "sequence_score" in metrics_list or len(metrics_list) == 0:
            threshold = datamodule.test_datasets[0].threshold
            g_sequence_score_results, g_scores, scanpath_recall, covered_human_scanpaths, g_scores_by_key = compute_SS_Time(
                modified_generations,
                clusters=ss_clusters,
                threshold=threshold,
                truncate=16,
                time_dict=t_dict,
                is_validation=is_validation,
                return_scores_by_key=True,
            ) #review the truncate value
            ss_final_score = np.mean([r['score'] for r in g_sequence_score_results])
            metrics.loc["Model", "SequenceScoreTime"] = ss_final_score
            metrics.loc["Model", "SequenceScoreTimeBest"] = _mean_of_per_gt_bests(g_scores_by_key, num_gt_by_key, num_gen_by_key, higher_is_better=True)
            metrics.loc["Scanpath_Recall", "SequenceScoreTime"] = scanpath_recall
            
            h_sequence_score_results, h_scores = compute_self_SS_Time(originals=originals, clusters=ss_clusters, 
                                                                    truncate=16, time_dict=t_dict, is_validation=is_validation) #review the truncate value
            ss_final_score = np.mean([r['score'] for r in h_sequence_score_results])
            metrics.loc["Human", "SequenceScoreTime"] = ss_final_score  # type: ignore
            
            if "kld" in metrics_list or len(metrics_list) == 0:
                print("Calculating KDL in Sequence Score Time...")
                metrics.loc["KLD", "SequenceScoreTime"] = perform_kld(h_scores, g_scores)

            if "crps" in metrics_list or len(metrics_list) == 0:
                print("Calculating CRPS in Sequence Score Time...")
                generated_by_condition_key = {}
                for scanpath in modified_generations:
                    condition_key = _sequence_condition_key(scanpath, is_validation=is_validation)
                    generated_by_condition_key.setdefault(condition_key, []).append(scanpath)

                num_gen_by_key = {
                    key: len(scanpaths)
                    for key, scanpaths in generated_by_condition_key.items()
                }
                num_gt_by_key = {
                    key: len(ss_clusters[key]["strings"])
                    for key in g_scores_by_key.keys()
                    if key in ss_clusters
                }

                g2g_scores_by_key = {}
                for key, scanpaths in generated_by_condition_key.items():
                    if len(scanpaths) < 2:
                        continue
                    g2g_scores_by_key[key] = compute_SS_Time_model_vs_model(
                        scanpaths,
                        clusters=ss_clusters,
                        truncate=16,
                        time_dict=t_dict,
                        threshold=threshold,
                        is_validation=is_validation,
                    )

                g2g_ordered_by_key = _expand_symmetric_combinations_to_ordered_with_self(
                    pair_scores_by_key=g2g_scores_by_key,
                    num_gen_by_key=num_gen_by_key,
                    self_score=1.0,
                )
                per_gt_scores_by_key = reshape_flat_scores_by_key(
                    flat_scores_by_key=g_scores_by_key,
                    num_gt_by_key=num_gt_by_key,
                    num_gen_by_key=num_gen_by_key,
                    generated_major=True,
                )
                metrics.loc["CRPS", "SequenceScoreTime"] = perform_crps(
                    per_gt_to_gen_scores_by_key=per_gt_scores_by_key,
                    gen_pair_scores_by_key=g2g_ordered_by_key,
                    convert_higher_better=False,
                    conversion_mode=None,
                )
                
        if "diversity_sequence_score" in metrics_list or len(metrics_list) == 0:
            print('Generating Model Diversity Sequence Score metrics...')
            threshold = datamodule.test_datasets[0].threshold
            DSS, DSS_per_img, all_human_vs_human, all_human_vs_model, all_model_vs_model = compute_DSS_Time(modified_generations, clusters=ss_clusters, threshold=threshold,
                                                                truncate=16, time_dict=t_dict, is_validation=is_validation)
            metrics.loc["Model", "DSSTime"] = DSS
            
        print('Done!')
        
    # SemSS segmentation maps live under the test dataset root (COCO-Search18 only).
    segmentation_maps_dir = str(Path(data_root or ".", "SemSS", "segmentation_maps"))

    semantic_requested = "semantic_sequence_score" in metrics_list or len(metrics_list) == 0
    skip_semantic_for_freeview = semantic_requested and any(
        dataset_name in freeview_datasets for dataset_name in dataset_names
    )
    semss_missing = semantic_requested and not Path(segmentation_maps_dir).exists()

    if skip_semantic_for_freeview or semss_missing:
        skip_reason = []
        if skip_semantic_for_freeview:
            skip_reason.append(
                f"freeview dataset detected ({sorted(dataset_names & freeview_datasets)})"
            )
        if semss_missing:
            skip_reason.append(f"missing SemSS directory: {segmentation_maps_dir}")
        print(
            "Skipping semantic sequence metrics and reporting NaN because "
            + " and ".join(skip_reason)
            + "."
        )
        metrics.loc["Model", "SemanticSequenceScore"] = np.nan  # type: ignore
        metrics.loc["Model", "SemanticSequenceScoreBest"] = np.nan  # type: ignore
        metrics.loc["Human", "SemanticSequenceScore"] = np.nan  # type: ignore
        metrics.loc["KLD", "SemanticSequenceScore"] = np.nan  # type: ignore
        metrics.loc["CRPS", "SemanticSequenceScore"] = np.nan  # type: ignore
        metrics.loc["Model", "SemanticSequenceScoreTime"] = np.nan  # type: ignore
        metrics.loc["Model", "SemanticSequenceScoreTimeBest"] = np.nan  # type: ignore
        metrics.loc["Human", "SemanticSequenceScoreTime"] = np.nan  # type: ignore
        metrics.loc["KLD", "SemanticSequenceScoreTime"] = np.nan  # type: ignore
        metrics.loc["CRPS", "SemanticSequenceScoreTime"] = np.nan  # type: ignore

    if ("semantic_sequence_score" in metrics_list or len(metrics_list) == 0) and not (
        skip_semantic_for_freeview or semss_missing
    ):
        #compute semantic sequence score
        print('Generating Model Semantic Sequence Score metrics...')
        # adjust predictions for SSS computation
        modified_generations = []
        for key, value in tqdm(generated.items()):
            for sample in value['scanpaths']:
                scanpath = {}
                
                if isinstance(key, tuple):
                    scanpath['name'] = str(Path(key[0]).stem)
                    scanpath['task'] = key[1]
                else:
                    scanpath['name'] = str(Path(key).stem)
                
                scanpath['dataset'] = value['dataset']
                
                if scanpath['dataset'] in ['cocosearch18_tp', 'cocosearch18_edf_tp']:
                    scanpath['condition'] = 'present'
                elif scanpath['dataset'] in ['cocosearch18_ta', 'cocosearch18_edf_ta']:
                    scanpath['condition'] = 'absent'
                else:
                    scanpath['condition'] = ''
                
                scanpath['question_id'] = value['question_id']

                scanpath['X'] = sample['X']
                scanpath['Y'] = sample['Y']
                if 'T' in sample:
                    scanpath['T'] = sample['T']
                modified_generations.append(scanpath)
                
        modified_originals = []
        for key, value in tqdm(originals.items()):
            for sample in value['scanpaths']:
                scanpath = {}
                
                if isinstance(key, tuple):
                    scanpath['name'] = str(Path(key[0]).stem)
                    scanpath['task'] = key[1]
                else:
                    scanpath['name'] = str(Path(key).stem)
                
                scanpath['dataset'] = value['dataset']
                
                if scanpath['dataset'] in ['cocosearch18_tp', 'cocosearch18_edf_tp']:
                    scanpath['condition'] = 'present'
                elif scanpath['dataset'] in ['cocosearch18_ta', 'cocosearch18_edf_ta']:
                    scanpath['condition'] = 'absent'
                else:
                    scanpath['condition'] = ''
                
                scanpath['question_id'] = value['question_id']

                scanpath['X'] = sample['X']
                scanpath['Y'] = sample['Y']
                if 'T' in sample:
                    scanpath['T'] = sample['T']
                modified_originals.append(scanpath)
        
        g_sequence_score_results, g_scores, g_scores_by_key = compute_SSS(
            modified_generations,
            originals,
            segmentation_maps_dir,
            truncate=16,
            is_validation=is_validation,
            return_scores_by_key=True,
        )
        sss_final_score = np.mean([r['score'] for r in g_sequence_score_results])
        metrics.loc["Model", "SemanticSequenceScore"] = sss_final_score
        metrics.loc["Model", "SemanticSequenceScoreBest"] = _mean_of_per_gt_bests(g_scores_by_key, num_gt_by_key, num_gen_by_key, higher_is_better=True)
        
        h_sequence_score_results, h_scores = compute_self_SSS(modified_originals, originals, segmentation_maps_dir, truncate=16, is_validation=is_validation)
        ss_final_score = np.mean([r['score'] for r in h_sequence_score_results])
        metrics.loc["Human", "SemanticSequenceScore"] = ss_final_score
        
        print(metrics)
        
        if "kld" in metrics_list or len(metrics_list) == 0:
            print("Calculating KDL in Semantic Sequence Score...")
            metrics.loc["KLD", "SemanticSequenceScore"] = perform_kld(h_scores, g_scores)

        if "crps" in metrics_list or len(metrics_list) == 0:
            print("Calculating CRPS in Semantic Sequence Score...")
            _, generated_generated_scores, generated_generated_scores_by_key = compute_SSS(
                modified_generations,
                generated,
                segmentation_maps_dir,
                truncate=16,
                is_validation=is_validation,
                return_scores_by_key=True,
            )
            _ = generated_generated_scores
            num_gt_by_key = {
                key: len(value["scanpaths"])
                for key, value in originals.items()
            }
            num_gen_by_key = {
                (scanpath["name"] + ".jpg", scanpath["task"].replace(" ", "_")): 0
                for scanpath in modified_generations
            }
            for scanpath in modified_generations:
                condition_key = (scanpath["name"] + ".jpg", scanpath["task"].replace(" ", "_"))
                num_gen_by_key[condition_key] += 1

            per_gt_scores_by_key = reshape_flat_scores_by_key(
                flat_scores_by_key=g_scores_by_key,
                num_gt_by_key=num_gt_by_key,
                num_gen_by_key=num_gen_by_key,
                generated_major=True,
            )
            metrics.loc["CRPS", "SemanticSequenceScore"] = perform_crps(
                per_gt_to_gen_scores_by_key=per_gt_scores_by_key,
                gen_pair_scores_by_key=generated_generated_scores_by_key,
                convert_higher_better=False,
                conversion_mode=None,
            )
        
        print('Done!')
    
    if ("semantic_sequence_score" in metrics_list or len(metrics_list) == 0) and not (
        skip_semantic_for_freeview or semss_missing
    ):
        #compute semantic sequence score time
        print('Generating Model Semantic Sequence Score Time metrics...')
        # adjust predictions for SSSTime computation
        modified_generations = []
        for key, value in tqdm(generated.items()):
            for sample in value['scanpaths']:
                scanpath = {}
                
                if isinstance(key, tuple):
                    scanpath['name'] = str(Path(key[0]).stem)
                    scanpath['task'] = key[1]
                else:
                    scanpath['name'] = str(Path(key).stem)
                
                scanpath['dataset'] = value['dataset']
                
                if scanpath['dataset'] in ['cocosearch18_tp', 'cocosearch18_edf_tp']:
                    scanpath['condition'] = 'present'
                elif scanpath['dataset'] in ['cocosearch18_ta', 'cocosearch18_edf_ta']:
                    scanpath['condition'] = 'absent'
                else:
                    scanpath['condition'] = ''
                
                scanpath['question_id'] = value['question_id']

                scanpath['X'] = sample['X']
                scanpath['Y'] = sample['Y']
                if 'T' in sample:
                    scanpath['T'] = sample['T']
                modified_generations.append(scanpath)
                
        modified_originals = []
        for key, value in tqdm(originals.items()):
            for sample in value['scanpaths']:
                scanpath = {}
                
                if isinstance(key, tuple):
                    scanpath['name'] = str(Path(key[0]).stem)
                    scanpath['task'] = key[1]
                else:
                    scanpath['name'] = str(Path(key).stem)
                
                scanpath['dataset'] = value['dataset']
                
                if scanpath['dataset'] in ['cocosearch18_tp', 'cocosearch18_edf_tp']:
                    scanpath['condition'] = 'present'
                elif scanpath['dataset'] in ['cocosearch18_ta', 'cocosearch18_edf_ta']:
                    scanpath['condition'] = 'absent'
                else:
                    scanpath['condition'] = ''
                
                scanpath['question_id'] = value['question_id']

                scanpath['X'] = sample['X']
                scanpath['Y'] = sample['Y']
                if 'T' in sample:
                    scanpath['T'] = sample['T']
                modified_originals.append(scanpath)
        
        g_sequence_score_results, g_scores, g_scores_by_key = compute_SSS_Time(
            modified_generations,
            originals,
            segmentation_maps_dir,
            truncate=16,
            is_validation=is_validation,
            return_scores_by_key=True,
        )
        sss_final_score = np.mean([r['score'] for r in g_sequence_score_results])
        metrics.loc["Model", "SemanticSequenceScoreTime"] = sss_final_score
        metrics.loc["Model", "SemanticSequenceScoreTimeBest"] = _mean_of_per_gt_bests(g_scores_by_key, num_gt_by_key, num_gen_by_key, higher_is_better=True)

        h_sequence_score_results, h_scores = compute_self_SSS_Time(modified_originals, originals, segmentation_maps_dir, truncate=16, is_validation=is_validation)
        sss_final_score = np.mean([r['score'] for r in h_sequence_score_results])
        metrics.loc["Human", "SemanticSequenceScoreTime"] = sss_final_score
        
        print(metrics)
        
        if "kld" in metrics_list or len(metrics_list) == 0:
            print("Calculating KDL in Semantic Sequence Score TIme...")
            metrics.loc["KLD", "SemanticSequenceScoreTime"] = perform_kld(h_scores, g_scores)

        if "crps" in metrics_list or len(metrics_list) == 0:
            print("Calculating CRPS in Semantic Sequence Score Time...")
            _, generated_generated_scores, generated_generated_scores_by_key = compute_SSS_Time(
                modified_generations,
                generated,
                segmentation_maps_dir,
                truncate=16,
                is_validation=is_validation,
                return_scores_by_key=True,
            )
            _ = generated_generated_scores
            num_gt_by_key = {
                key: len(value["scanpaths"])
                for key, value in originals.items()
            }
            num_gen_by_key = {
                (scanpath["name"] + ".jpg", scanpath["task"].replace(" ", "_")): 0
                for scanpath in modified_generations
            }
            for scanpath in modified_generations:
                condition_key = (scanpath["name"] + ".jpg", scanpath["task"].replace(" ", "_"))
                num_gen_by_key[condition_key] += 1

            per_gt_scores_by_key = reshape_flat_scores_by_key(
                flat_scores_by_key=g_scores_by_key,
                num_gt_by_key=num_gt_by_key,
                num_gen_by_key=num_gen_by_key,
                generated_major=True,
            )
            metrics.loc["CRPS", "SemanticSequenceScoreTime"] = perform_crps(
                per_gt_to_gen_scores_by_key=per_gt_scores_by_key,
                gen_pair_scores_by_key=generated_generated_scores_by_key,
                convert_higher_better=False,
                conversion_mode=None,
            )
        
        print('Done!')


    if 'conditional_information_gain' in metrics_list or len(metrics_list) == 0:
        pass  
    
    if "cc" in metrics_list or "nss" in metrics_list or "auc" in metrics_list or "kldiv" in metrics_list or len(metrics_list) == 0:
        print("Generating Model Saliency Metrics...")
        print('Metrics list is: ', metrics_list)
        nss, cc, auc, kldiv = compute_saliency_metrics(
            originals, generated, (images_width, images_height)
        )
        if "nss" in metrics_list or len(metrics_list) == 0:
            metrics.loc["Model", "NSS"] = nss  # type: ignore
        if "cc" in metrics_list or len(metrics_list) == 0:
            metrics.loc["Model", "CC"] = cc  # type: ignore
        if "auc" in metrics_list or len(metrics_list) == 0:
            metrics.loc["Model", "AUC"] = auc
        if "kldiv" in metrics_list or len(metrics_list) == 0:
            metrics.loc["Model", "KLDiv"] = kldiv
        print("Done.")

    print('The final metrics are:')
    print(DSS_per_img)
    print(DSS_per_img_no_dur)
    print(covered_human_scanpaths_no_dur)
    print(covered_human_scanpaths)
    print(all_human_vs_human)
    print(all_human_vs_model)
    print(all_model_vs_model)
    
    return metrics, DSS_per_img, DSS_per_img_no_dur, covered_human_scanpaths_no_dur, covered_human_scanpaths, all_human_vs_human, all_human_vs_model, all_model_vs_model
