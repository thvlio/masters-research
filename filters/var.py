import datetime
from pathlib import Path

import matplotlib
import matplotlib.axes
import matplotlib.figure
import numpy as np
import pandas as pd
import scipy.fft
import scipy.io
import scipy.signal
from matplotlib import pyplot as plt


def dft(signal, sampling_freq, num_samples):
    freqs = scipy.fft.rfftfreq(num_samples, 1 / sampling_freq)
    spectrum = scipy.fft.rfft(signal, n=num_samples)
    mags = 2 * np.abs(spectrum) / num_samples
    # mags = np.angle(spectrum, True)
    return freqs, mags


def pgram(timestamps, signal, min_freq, max_freq, num_samples):
    freqs = np.linspace(min_freq, max_freq, num_samples)
    pgram = scipy.signal.lombscargle(timestamps, signal, freqs=freqs*2*np.pi, floating_mean=True, normalize=True)
    mags = pgram
    return freqs, mags


def main():

    plt.rcParams.update({'font.size': 22})

    signal_path = Path('/home/thulio/Downloads/ppgeb/3314767_0004m.mat')
    header_path = signal_path.with_suffix('.hea')

    header_lines = header_path.read_text().split('\n')
    header = header_lines[0].strip().split()
    specs = [li.strip().split() for li in header_lines[1:] if not li.startswith('#') and len(li.strip()) > 0]

    signal_info = []
    num_signals = int(header[1])
    sampling_freq = float(header[2])
    num_samples = int(header[3])
    start_time = datetime.datetime.strptime(header[4]+'000', '%H:%M:%S.%f')
    # start_time = datetime.datetime.fromisoformat(header[4])

    timestamps = np.linspace(start_time, start_time + datetime.timedelta(seconds=num_samples/sampling_freq), num_samples)
    signals = scipy.io.loadmat(signal_path)

    for ss in range(num_signals):
        name = specs[ss][-1]
        gain_base, unit = specs[ss][2].split('/')
        gain, base = map(float, gain_base.replace('(', ' ').replace(')', ' ').strip().split())
        signal_info.append({'name': name, 'gain': gain, 'base': base, 'unit': unit})

    df_info = pd.DataFrame(signal_info).set_index('name', drop=True)
    df_signals = pd.DataFrame({info['name']: data for info, data in zip(signal_info, signals['val'])}, index=timestamps)

    ax_t: matplotlib.axes.Axes
    ax_s: matplotlib.axes.Axes
    ax_v: matplotlib.axes.Axes
    ax_w: matplotlib.axes.Axes
    fig_ppg, ((ax_t, ax_v), (ax_s, ax_w)) = plt.subplots(2, 2, figsize=(16, 16))
    alpha = 1.0

    fs = sampling_freq
    min_freq = 0.8
    max_freq = 4.0
    butter_order = 16
    butter_bands = [min_freq, max_freq]
    detrend_method = 'linear'

    signal_name = 'PLETH'

    ppg = (df_signals[signal_name] - df_info.loc[signal_name]['base']) / df_info.loc[signal_name]['gain']
    ax_t.plot(timestamps, ppg, alpha=alpha)
    freqs, mags = dft(ppg, sampling_freq, num_samples)
    ax_s.plot(freqs, mags, alpha=alpha)

    ppg_d = scipy.signal.detrend(ppg, type=detrend_method)
    butter = scipy.signal.butter(butter_order, butter_bands, btype='bandpass', output='sos', fs=fs)
    ppg_f = scipy.signal.sosfiltfilt(butter, ppg_d)
    ax_t.plot(timestamps, ppg_f, alpha=alpha)
    freqs, mags = dft(ppg_f, sampling_freq, num_samples)
    ax_s.plot(freqs, mags, alpha=alpha)

    peaks, _ = scipy.signal.find_peaks(ppg, prominence=0.25)
    ax_t.scatter(timestamps[peaks], ppg.iloc[peaks])

    t = (timestamps[peaks] - timestamps[peaks][0]).astype(np.timedelta64) / np.timedelta64(1, 's')
    v = np.diff(t, prepend=t[0])
    ax_v.plot(timestamps[peaks], v, alpha=alpha)
    # freqs, mags = dft(v, sampling_freq, num_samples)
    freqs, mags = pgram(t, v, min_freq, sampling_freq/10, num_samples)
    ax_w.plot(freqs, mags, alpha=alpha)

    v_d = scipy.signal.detrend(v, type=detrend_method)
    # butter = scipy.signal.butter(butter_order, butter_bands, btype='bandpass', output='sos', fs=fs)
    # v_f = scipy.signal.sosfiltfilt(butter, v_d)
    v_f = v_d
    ax_v.plot(timestamps[peaks], v_f, alpha=alpha)
    # freqs, mags = dft(v_f, sampling_freq, num_samples)
    freqs, mags = pgram(t, v_f, min_freq, sampling_freq/10, num_samples)
    ax_w.plot(freqs, mags, alpha=alpha)

    ax_t.set(title=(lt := f'{signal_name} ({df_info.loc[signal_name]['unit']})'), xlabel='Time', ylabel=lt)
    ax_t.legend(['raw', 'processed', 'peaks'])
    ax_t.grid(True)
    ax_s.set(xlabel='Frequency', ylabel='Magnitude', xlim=[0.01, sampling_freq/10], ylim=[0.0, 0.2])
    # ax_s.set(xlabel='Frequency', ylabel='Phase')
    ax_s.legend(['raw', 'processed'])
    ax_s.grid(True)
    ax_v.grid(True)
    ax_w.set(xlabel='Frequency', ylabel='Magnitude', xlim=[0.01, sampling_freq/10], ylim=[0.0, 0.05])
    ax_w.grid(True)
    fig_ppg.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
