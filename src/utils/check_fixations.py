"""
Python implementation of checkFixations.m
Identifies Saccades and Fixations from eye tracking data.

Converted from MATLAB code by Barbara Hidalgo-Sotelo (12/18/05, updated 6/28/06)
"""

import numpy as np
from scipy.signal import lfilter
from dataclasses import dataclass
from typing import Tuple, Optional

import sys

# Lower means more sensitive saccade detection; higher means fewer detected saccades. Default 6.0
DEFAULT_DEG_PER_SEC = 6.0
# Minimum fixation duration in milliseconds; lower keeps more short fixations. Default 50
DEFAULT_FIX_DUR_CUT_MS = 15
# Max forward-search steps when looking for deceleration after acceleration crossing. Default 25
DEFAULT_DECEL_SEARCH_STEPS = 25
# Smoothing and temporal settings used by the original algorithm.
DEFAULT_WIN_SIZE = 4
DEFAULT_STEP_SIZE = 1
# Sample rate of data, MIT1003 is 240 Hz. COCOSearch18 is 1000 Hz
DEFAULT_SAMPLING_RATE = 240

@dataclass
class FixationData:
    """Container for fixation information"""
    duration: np.ndarray  # Duration of each fixation in ms
    medianXY: np.ndarray  # Median X,Y coordinates of each fixation
    start: np.ndarray  # Start index of each fixation
    end: np.ndarray  # End index of each fixation
    driftXY: np.ndarray  # Variance in X,Y during each fixation
    badEyeTrack: bool  # Flag for bad eye tracking data
    badNaN: Optional[bool]  # Flag for excessive NaNs


@dataclass
class SaccadeData:
    """Container for saccade information"""
    duration: np.ndarray  # Duration of each saccade in ms
    distPx: np.ndarray  # Distance of saccade in pixels
    distDeg: np.ndarray  # Distance of saccade in degrees
    badEyeTrack: bool  # Flag for bad eye tracking data
    badNaN: Optional[bool]  # Flag for excessive NaNs


def check_fixations(data: np.ndarray, allowed_nans: Optional[int] = None, debug: bool = False) -> Tuple[np.ndarray, FixationData, SaccadeData]:
    """
    Identify saccades and fixations from eye tracking data.
    
    Args:
        data: Nx2 array with columns [POR_X, POR_Y] (Point of Regard X and Y coordinates)
        allowed_nans: Optional threshold for number of allowable NaNs
        debug: If True, print detailed segmentation info
    
    Returns:
        data: Nx3 array with third column indicating:
            0 = fixation
            1 = middle of saccade
            2 = end of saccade
            3 = start of saccade
        fix_data: FixationData object with fixation information
        sac_data: SaccadeData object with saccade information
    """
    
    # PARAMETERS
    deg_per_sec = DEFAULT_DEG_PER_SEC  # Acceleration threshold
    px_per_deg = 34.0  # based on ET setup (display=1024x768px=40.5x30cm; distance=75cm)
    accel_crit = px_per_deg * deg_per_sec  # 204 px/sec
    win_size = DEFAULT_WIN_SIZE  # 4 samples = 16.667 ms
    step_size = DEFAULT_STEP_SIZE  # 1 sample = 4.167 ms
    sampling_rate = DEFAULT_SAMPLING_RATE  # Hz
    
    fix_dur_cut = DEFAULT_FIX_DUR_CUT_MS  # Shortest fixation allowed [ms]
    fix_dur_cut_pts = int(np.round(fix_dur_cut * sampling_rate / 1000))
    
    bad_et_data_threshold = 0.5  # Max percent of trial = NaN
    
    # Quality Control
    count_nan = np.sum(np.isnan(data[:, 0]))
    percent_nan = count_nan / len(data)
    
    bad_eye_track_flag = percent_nan > bad_et_data_threshold
    
    if allowed_nans is not None:
        bad_nan_flag = count_nan > allowed_nans
    else:
        bad_nan_flag = None
    
    # Add third column for fixation/saccade labels
    data = np.column_stack([data, np.zeros(len(data))])
    
    # Calculate distance between successive data points (in px)
    dx = np.diff(data[:, 0])
    dy = np.diff(data[:, 1])
    pos_change = np.sqrt(dx**2 + dy**2)
    pos_change = np.concatenate([[0], pos_change])
    
    # Match MATLAB filter(ones(1,winSize)/winSize, 1, posChange): causal moving average
    pos_change_sm = lfilter(np.ones(win_size) / win_size, 1, pos_change)
    
    # Find acceleration of 2 overlapping windows
    vel_win = np.zeros((len(pos_change), 2))
    for k in range(1, len(pos_change) - win_size - step_size):
        vel_win[k, 0] = np.sum(pos_change_sm[k:k+win_size]) / (win_size / sampling_rate)
        vel_win[k, 1] = np.sum(pos_change_sm[k+step_size:k+win_size+step_size]) / (win_size / sampling_rate)
    
    accel = vel_win[:, 1] - vel_win[:, 0]
    
    # Main saccade detection loop
    no_sac = True
    
    while no_sac:
        # Identify samples exceeding acceleration criterion
        check_sac = np.where(accel > accel_crit)[0]
        
        if len(check_sac) < 3:
            break
        
        # Index checkSac with markers for each saccade
        diff_check_sac = np.diff(check_sac)
        eachsac1 = np.where(diff_check_sac == 1)[0]  # successive points = same saccade
        eachsac2 = np.where(diff_check_sac > 1)[0]  # jump = end of saccade
        
        # Find "lone" points
        if len(eachsac2) > 1:
            diff_eachsac2 = np.diff(eachsac2)
            eachsac3_idx = np.where(diff_eachsac2 == 1)[0]
            eachsac3 = eachsac2[eachsac3_idx + 1]
        else:
            eachsac3 = np.array([])
        
        # Handle last row in checkSac
        if len(check_sac) > 1:
            if check_sac[-1] - check_sac[-2] == 1:
                eachsac1 = np.append(eachsac1, len(check_sac) - 1)
            else:
                eachsac3 = np.append(eachsac3, len(check_sac) - 1)
        
        # Create indexing array
        n = np.zeros((len(check_sac), 2), dtype=int)
        n[:, 0] = check_sac
        n[eachsac1, 1] = 1  # middle of saccade
        n[eachsac2, 1] = 2  # end of acceleration part
        n[eachsac3.astype(int), 1] = 3  # lone points
        
        # Handle last row
        if len(check_sac) > 1 and check_sac[-1] - check_sac[-2] == 1:
            n[-1, 1] = 2
        else:
            n = n[:-1, :]
            eachsac2 = eachsac2[:-1] if len(eachsac2) > 0 else eachsac2
        
        # Mark beginning of saccades
        eachsac3 = eachsac2 + 1
        n[eachsac3, 1] = 3
        
        # Check first data point
        if len(n) > 1 and n[0, 1] == 1:
            n[0, 1] = 3
        
        # Identify contiguous saccades, discard isolated points
        nn = []
        if len(n) > 1 and n[0, 1] == 3 and n[1, 1] == 1:
            nn.append(n[0])
        
        for i in range(1, len(n) - 1):
            if n[i, 1] == 3 and n[i+1, 1] == 1:
                nn.append(n[i])
            elif n[i, 1] == 1:
                nn.append(n[i])
            elif n[i, 1] == 2 and n[i-1, 1] == 1 and n[i+1, 1] == 3:
                nn.append(n[i])
        
        # Check last element
        if len(n) > 1 and len(nn) > 0:
            if n[-1, 0] - nn[-1][0] == 1:
                nn.append(n[-1])
        
        check_sac = np.array(nn) if nn else np.array([])
        
        if len(check_sac) < 3:
            break
        
        # Use deceleration criteria to identify end of each saccade
        eachsac2 = np.where(check_sac[:, 1] == 2)[0]
        eachsac3 = np.where(check_sac[:, 1] == 3)[0]
        
        real_sac = 0
        good_sac = []
        blah = []
        
        for sac in range(len(eachsac2)):
            check = np.array([check_sac[eachsac2[sac], 0] + 1, check_sac[eachsac2[sac], 0] + 2])
            crosscheck = accel[check] < -accel_crit
            stop_after = DEFAULT_DECEL_SEARCH_STEPS
            time_ok = True
            
            while not (crosscheck[0] and not crosscheck[1]) and time_ok:
                check = check + 1
                if np.max(check) >= len(data):
                    time_ok = False
                    break
                crosscheck = accel[check] < -accel_crit
                stop_after -= 1
                time_ok = (np.max(check) < len(data)) and (stop_after > 0)
            
            # No deceleration found
            if not time_ok:
                # Unless trial ends in saccade
                if check_sac[eachsac2[sac], 0] == np.where(accel != 0)[0][-1]:
                    good_sac.append(sac)
                    real_sac += 1
                    blah.append(len(accel) - 1)
            else:
                # Deceleration found
                good_sac.append(sac)
                real_sac += 1
                blah.append(check[0])
        
        if len(good_sac) == 0 or len(check_sac) == 0:
            break
        
        good_sac = np.array(good_sac)
        good_sac2 = eachsac2[good_sac]
        good_sac3 = eachsac3[good_sac]
        decel = np.column_stack([check_sac[good_sac2, 0], blah])
        
        # Check fixation duration cutoff
        if real_sac > 1:
            too_short = []
            for i in range(len(blah) - 1):
                if check_sac[good_sac3[i+1], 0] - decel[i, 1] <= fix_dur_cut_pts:
                    too_short.append(i + 1)
            
            keep_start = list(set(range(len(blah))) - set([x for x in too_short]))
            keep_end = list(set(range(len(blah))) - set([x - 1 for x in too_short]))
            
            start_sac = check_sac[good_sac3[keep_start], 0]
            end_sac = decel[keep_end, 1]
        else:
            start_sac = check_sac[good_sac3, 0]
            end_sac = decel[:, 1]
        
        # Check final fixation length
        if len(data) - decel[-1, 1] <= fix_dur_cut_pts:
            end_sac[-1] = len(data) - 1
        
        # Label saccades/fixations
        eachsac3 = start_sac
        eachsac2 = end_sac
        eachsac1 = []
        for f in range(len(eachsac3)):
            eachsac1.extend(range(eachsac3[f] + 1, eachsac2[f]))
        
        data[:, 2] = 0
        data[eachsac3, 2] = 3
        data[eachsac2, 2] = 2
        data[eachsac1, 2] = 1
        
        # Find fixation start/end
        check_fixs = np.where(data[:, 2] == 0)[0]
        check_sac = np.setdiff1d(np.arange(len(data)), check_fixs)
        
        if len(check_sac) < 3:
            break
        
        eachsac3 = np.where(data[:, 2] == 3)[0]
        eachsac2 = np.where(data[:, 2] == 2)[0]
        
        # Determine fixation start positions
        case1 = (eachsac3[0] == 0)  # start in saccade
        case2 = (eachsac2[-1] == len(data) - 1)  # end in saccade
        
        if len(eachsac3) == 1:
            if not case2:
                fix_start = np.array([check_fixs[0], eachsac2[0] + 1])
            else:
                fix_start = np.array([check_fixs[0]])
        else:
            # Match MATLAB: always seed with first fixation sample, regardless of case1.
            fix_start = [check_fixs[0]] + (eachsac2[:-1] + 1).tolist()
            if not case2:
                fix_start.append(eachsac2[-1] + 1)
            fix_start = np.array(fix_start)
        
        # Determine fixation end positions
        if not case1 and not case2:
            fix_end = np.append(eachsac3 - 1, len(data) - 1)
        elif case1 and not case2:
            fix_end = np.append(eachsac3[1:] - 1, len(data) - 1)
        elif not case1 and case2:
            fix_end = eachsac3 - 1
        else:  # case1 and case2
            fix_end = eachsac3[1:] - 1
        
        # Get fixation info
        n_fixs = len(fix_start)
        n_sacs = len(eachsac3)
        
        fix_duration = np.zeros(n_fixs)
        fix_median_xy = np.zeros((n_fixs, 2))
        fix_drift_xy = np.zeros((n_fixs, 2))
        
        for i in range(n_fixs):
            fix_duration[i] = 1000 * (fix_end[i] - fix_start[i] + 1) / sampling_rate
            
            tmp_x = data[fix_start[i]:fix_end[i]+1, 0]
            tmp_x_clean = tmp_x[~np.isnan(tmp_x)]
            fix_median_xy[i, 0] = np.median(tmp_x_clean) if len(tmp_x_clean) > 0 else np.nan
            
            tmp_y = data[fix_start[i]:fix_end[i]+1, 1]
            tmp_y_clean = tmp_y[~np.isnan(tmp_y)]
            fix_median_xy[i, 1] = np.median(tmp_y_clean) if len(tmp_y_clean) > 0 else np.nan
            
            fix_drift_xy[i, 0] = np.var(tmp_x_clean) if len(tmp_x_clean) > 0 else np.nan
            fix_drift_xy[i, 1] = np.var(tmp_y_clean) if len(tmp_y_clean) > 0 else np.nan
        
        if debug:
            print(f"DEBUG: Detected {real_sac} saccades", file=sys.stderr)
            print(f"DEBUG: Fixation starts: {fix_start}", file=sys.stderr)
            print(f"DEBUG: Fixation ends: {fix_end}", file=sys.stderr)
            for i in range(n_fixs):
                seg_len = fix_end[i] - fix_start[i] + 1
                median = fix_median_xy[i]
                print(f"DEBUG:   Fix {i}: samples[{fix_start[i]}:{fix_end[i]}] len={seg_len} median=({median[0]:.1f}, {median[1]:.1f})", file=sys.stderr)
        fix_data = FixationData(
            duration=fix_duration,
            medianXY=fix_median_xy,
            start=fix_start,
            end=fix_end,
            driftXY=fix_drift_xy,
            badEyeTrack=bad_eye_track_flag,
            badNaN=bad_nan_flag
        )
        
        sac_duration = 1000 * (eachsac2 - eachsac3 + 1) / sampling_rate
        sac_dist_px = np.sqrt(np.diff(fix_median_xy[:, 0])**2 + np.diff(fix_median_xy[:, 1])**2)
        sac_dist_deg = sac_dist_px / px_per_deg
        
        sac_data = SaccadeData(
            duration=sac_duration,
            distPx=sac_dist_px,
            distDeg=sac_dist_deg,
            badEyeTrack=bad_eye_track_flag,
            badNaN=bad_nan_flag
        )
        
        return data, fix_data, sac_data
    
    # If no saccades detected (whole trial is one fixation)
    n_sacs = 0
    n_fixs = 1
    
    fix_duration = np.array([1000 * len(data) / sampling_rate])
    fix_median_xy = np.array([[np.nanmedian(data[:, 0]), np.nanmedian(data[:, 1])]])
    fix_drift_xy = np.array([[np.nanvar(data[:, 0]), np.nanvar(data[:, 1])]])
    
    fix_data = FixationData(
        duration=fix_duration,
        medianXY=fix_median_xy,
        start=np.array([0]),
        end=np.array([len(data) - 1]),
        driftXY=fix_drift_xy,
        badEyeTrack=bad_eye_track_flag,
        badNaN=bad_nan_flag
    )
    
    sac_data = SaccadeData(
        duration=np.array([0]),
        distPx=np.array([0]),
        distDeg=np.array([0]),
        badEyeTrack=bad_eye_track_flag,
        badNaN=bad_nan_flag
    )
    
    data[:, 2] = 0
    
    return data, fix_data, sac_data
