import contextlib
import itertools
from pathlib import Path
# from typing import Literal

import neurokit2 as nk
import numpy as np
import pandas as pd
import pyPPG.fiducials as FP
import scipy.io
import scipy.signal
from dotmap import DotMap
from pyPPG import PPG, Fiducials

from config import Config as c


def load_subjects(
            subjects_path: str | Path = c.SUBJECTS_PATH
        ):
    return pd.read_csv(subjects_path)


def load_points(
            points_path: str | Path = c.POINTS_PATH
        ):

    df_points = pd.read_csv(
        points_path,
        dtype={
            'subject_id': pd.Int64Dtype(),
            'signal_name': pd.StringDtype(),
            'x': pd.Float64Dtype(),
            'y': pd.Float64Dtype(),
            'label': pd.Int64Dtype(),
        }
    )

    df_points = df_points.sort_values(['subject_id', 'signal_name', 'label', 'x'])
    df_points = df_points.set_index(['subject_id', 'signal_name', 'label'])

    return df_points


def get_peaks_and_onsets(
            df_points: pd.DataFrame,
            subject_id: int,
            *,
            fs: int = c.SIGNAL_FS,
            trim_peaks: bool = True
        ):

    peaks = (df_points.loc[(subject_id, 'PLETH', 1)] * fs)['x'].round().astype(int).to_numpy()
    onsets = (df_points.loc[(subject_id, 'PLETH', 3)] * fs)['x'].round().astype(int).to_numpy()

    if trim_peaks:
        peaks = peaks[(peaks > onsets[0]) & (peaks < onsets[-1])]

    cycles = np.array(list(itertools.pairwise(onsets)))

    return peaks, onsets, cycles


def load_ppg(
            subject_id: int,
            segment_id: str,
            *,
            ppg_dir: str | Path = c.PPG_DIR,
            s0: int | None = 2 * 60 * c.SIGNAL_FS,
            s1: int | None = 3 * 60 * c.SIGNAL_FS,
            correct_time: bool = True
        ) -> tuple[np.ndarray, np.ndarray]:

    patient_file = ppg_dir / f'p{subject_id:06d}_{segment_id}.mat'

    mat = scipy.io.loadmat(patient_file)

    t = mat['time'].squeeze()[s0:s1]
    ppg = mat['data'].squeeze()[s0:s1]

    if correct_time:
        t -= t[0]

    return t, ppg


def make_filter(
            design: str = 'butter',
            wn: int | tuple[int, int] = (0.5, 30),
            *,
            fs: int = c.SIGNAL_FS
        ):

    if design == 'cheby2':
        filt = scipy.signal.cheby2(4, 20, wn, btype='band' if isinstance(wn, (list, tuple)) else 'low', fs=fs, output='sos')
    elif design == 'butter':
        filt = scipy.signal.butter(4, wn, btype='band' if isinstance(wn, (list, tuple)) else 'low', fs=fs, output='sos')
    elif design == 'bessel':
        filt = scipy.signal.bessel(4, wn, btype='band' if isinstance(wn, (list, tuple)) else 'low', fs=fs, output='sos')
    else:
        raise NotImplementedError

    return filt


def filter_ppg(
            ppg: np.ndarray,
            filt: np.ndarray
        ):
    return scipy.signal.sosfiltfilt(filt, ppg)


def preprocess_ppg( # pylint: disable=W0102
            ppg_raw: np.ndarray,
            *,
            filtering: bool = True,
            ft_design: str = 'butter',
            ft_wn: int | tuple[int, int] = (0.5, 30),
            smoothing: bool = True,
            sm_ppg_only: bool = True,
            sm_wins: dict[str, int] = {'ppg': 50, 'vpg': 10, 'apg': 10, 'jpg': 10},
            savgol: bool = False,
            sg_win: int = 3,
            sg_poly: int = 2,
            fs: int = c.SIGNAL_FS
        ):

    if filtering:
        filt = make_filter(ft_design, ft_wn)
        ppg = filter_ppg(ppg_raw, filt)
    else:
        ppg = ppg_raw

    if smoothing:
        win = round(fs * sm_wins['ppg'] / 1000)
        b = 1 / win * np.ones(win)
        ppg = scipy.signal.filtfilt(b, 1, ppg)

    if savgol:
        vpg = scipy.signal.savgol_filter(ppg, sg_win, sg_poly, deriv=1)
    else:
        vpg = np.gradient(ppg)

    if smoothing and not sm_ppg_only:
        win = round(fs * sm_wins['vpg'] / 1000)
        b = 1 / win * np.ones(win)
        vpg = scipy.signal.filtfilt(b, 1, vpg)

    if savgol:
        apg = scipy.signal.savgol_filter(ppg, sg_win, sg_poly, deriv=2)
    else:
        apg = np.gradient(vpg)

    if smoothing and not sm_ppg_only:
        win = round(fs * sm_wins['apg'] / 1000)
        b = 1 / win * np.ones(win)
        apg = scipy.signal.filtfilt(b, 1, apg)

    if savgol:
        jpg = scipy.signal.savgol_filter(ppg, sg_win, sg_poly, deriv=3)
    else:
        jpg = np.gradient(apg)

    if smoothing and not sm_ppg_only:
        win = round(fs * sm_wins['jpg'] / 1000)
        b = 1 / win * np.ones(win)
        jpg = scipy.signal.filtfilt(b, 1, jpg)

    return ppg, vpg, apg, jpg


def normalize_cycles(
            ppg: np.ndarray[float],
            vpg: np.ndarray[float],
            apg: np.ndarray[float],
            jpg: np.ndarray[float],
            peaks: np.ndarray,
            cycles: pd.DataFrame,
            *,
            fs: int = c.SIGNAL_FS
        ):

    signals = (ppg, vpg, apg, jpg)
    normalized_signals = [np.array([scipy.signal.resample(signal[on:off], fs) for on, off in cycles]) for signal in signals]

    normalized_peaks = np.array([int(round((k + (peak - on) / (off - on)) * fs)) for k, ((on, off), peak) in enumerate(zip(cycles, peaks))])

    return normalized_signals, normalized_peaks


def select_cycles(
            ppg_norm_cycle_signals: np.ndarray,
            ppg_norm_cycle_peaks: np.ndarray | None = None,
            num_selected_cycles: int = c.NUM_SELECTED_CYCLES,
            num_discarded_edge_cycles: int = c.NUM_DISCARDED_EDGE_CYCLES,
            # method: Literal['template_match', 'skewness', 'random'] = 'template_match',
            fs: int = c.SIGNAL_FS,
        ):

    # if num_discarded_edge_cycles > 0:
    #     ppg_norm_cycle_signals = ppg_norm_cycle_signals[]

    # if method == 'random':

    ppg_norm_signal = ppg_norm_cycle_signals.flatten()

    # normalized_cycle_qualities = nk.ppg.ppg_quality(ppg_norm_signal, peaks=ppg_norm_cycle_peaks, sampling_rate=fs, method=quality_method)
    normalized_cycle_qualities = nk.ppg.ppg_quality(ppg_norm_signal, peaks=ppg_norm_cycle_peaks, sampling_rate=fs, method='templatematch')
    normalized_cycle_qualities = normalized_cycle_qualities.reshape(ppg_norm_cycle_signals.shape).mean(axis=1)

    sorted_indices = np.argsort(normalized_cycle_qualities)

    n = num_discarded_edge_cycles
    k = len(ppg_norm_cycle_signals)
    if (n := num_discarded_edge_cycles) > 0:
        sorted_indices = sorted_indices[(sorted_indices >= n) & (sorted_indices <= (k - n - 1))] if n > 0 else sorted_indices

    selected_indices = sorted_indices[::-1][:num_selected_cycles]

    selected_mask = np.zeros(ppg_norm_cycle_signals.shape[0], dtype=bool)
    selected_mask[selected_indices] = True

    return selected_indices, selected_mask


def get_fiducial_points( # pylint: disable=W0102
            ppg: np.ndarray[float],
            vpg: np.ndarray[float],
            apg: np.ndarray[float],
            jpg: np.ndarray[float],
            fs: int = c.SIGNAL_FS,
            correction: pd.DataFrame = pd.DataFrame(),
            peaks: list[int] | None = None,
            onsets: list[int] | None = None,
            check_ppg_len: bool = False
        ):

    s = DotMap(
        start_sig=0,
        end_sig=ppg.shape[0],
        # v=ppg_raw,
        ppg=ppg,
        vpg=vpg,
        apg=apg,
        jpg=jpg,
        fs=fs
        # name=subject_id+segment_id
    )

    corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
    correction.loc[0, corr_on] = True
    s.correction = correction

    # with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
    with contextlib.redirect_stderr(None):
        s = PPG(s=s, check_ppg_len=check_ppg_len)
        fpex = FP.FpCollection(s=s)
        fiducials = fpex.get_fiducials(s, peaks, onsets)
        fp = Fiducials(fp=fiducials)

    df_fps = fp.get_fp()

    return df_fps


def calc_indices(
            ppg: np.ndarray[float],
            _vpg: np.ndarray[float],
            apg: np.ndarray[float],
            _jpg: np.ndarray[float],
            s_fp: pd.Series,
            return_all: bool = False
        ):

    # replace missing dps with dns
    s_fp['dp'] = s_fp['dp'] if pd.notna(s_fp['dp']) else s_fp['dn']

    # ppg
    t_on = s_fp['on']
    s_z = ppg[t_on]
    t_sp = s_fp['sp']
    t_dp = s_fp['dp']
    a_s = ppg[t_sp] - s_z
    a_d = ppg[t_dp] - s_z
    s_50 = a_s / 2 + s_z
    t_50_rise = np.abs(ppg[:s_fp['sp']] - s_50).argmin()
    t_50_fall = np.abs(ppg[s_fp['sp']:] - s_50).argmin() + s_fp['sp']
    t_pw = t_50_fall - t_50_rise
    ri = a_d / a_s
    ai = (a_s - a_d) / a_s

    # vpg
    t_dt = t_dp - t_sp
    t_ct = t_sp - t_on

    # apg
    s_a = apg[s_fp['a']]
    s_b = apg[s_fp['b']]
    s_c = apg[s_fp['c']]
    s_d = apg[s_fp['d']]
    s_e = apg[s_fp['e']]
    r_b_a = s_b / s_a
    r_c_a = s_c / s_a
    r_d_a = s_d / s_a
    r_e_a = s_e / s_a
    r_b_c_d_e_a = (s_b - s_c - s_d - s_e) / s_a
    r_b_e_a = (s_b - s_e) / s_a
    r_b_c_d_a = (s_b - s_c - s_d) / s_a

    s_indices = pd.Series({
        # ppg
        'a_s': a_s,
        'a_d': a_d,
        't_pw': t_pw,
        'ri': ri,
        'ai': ai,
        # vpg
        't_dt': t_dt,
        't_ct': t_ct,
        # apg
        'r_b_a': r_b_a,
        'r_c_a': r_c_a,
        'r_d_a': r_d_a,
        'r_e_a': r_e_a,
        'r_b_c_d_e_a': r_b_c_d_e_a,
        'r_b_e_a': r_b_e_a,
        'r_b_c_d_a': r_b_c_d_a
    })

    if return_all:
        s_others = pd.Series({
            # ppg
            't_on': t_on,
            's_z': s_z,
            't_sp': t_sp,
            't_dp': t_dp,
            's_50': s_50,
            't_50_rise': t_50_rise,
            't_50_fall': t_50_fall,
            # apg
            's_a': s_a,
            's_b': s_b,
            's_c': s_c,
            's_d': s_d,
            's_e': s_e
        })
        s_indices = pd.concat([s_indices, s_others])

    return s_indices
