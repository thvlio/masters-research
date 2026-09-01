import logging
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.signal
import wfdb.io
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton, MouseEvent, PickEvent
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.widgets import Slider, Button

from config import Config as c


# setup logger

logging.basicConfig(format='[{asctime} {name} {levelname:8s}] {message}', style='{')
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# setup constants

DISPLAY_TIME_WINDOW = 2

DISPLAY_POINT_SIZE = 100

DISPLAY_POINT_COLOR_MAP = {
    MouseButton.LEFT: 'C9',
    MouseButton.RIGHT: 'C6',
    MouseButton.MIDDLE: 'C7'
}


def main():

    # setup figure and sliders

    fig, axes = plt.subplots(len(c.SIGNAL_NAMES), 1, figsize=(16, 8), sharex=True, layout='constrained', num='signal annotator')
    axes: tuple[Axes]

    fig.get_layout_engine().set(rect=(0.005, 0.12, 0.990, 0.875))

    plt.rcParams.update({'font.size': 18})

    # figure functions

    def plot_point(
                x: float,
                y: float,
                label: int,
                ax: Axes
            ) -> PathCollection:
        color = DISPLAY_POINT_COLOR_MAP[label]
        return ax.scatter(x, y, s=DISPLAY_POINT_SIZE, color=color, picker=True)

    def plot_points(
                subject_id: int
            ) -> None:
        for signal_name in c.SIGNAL_NAMES:
            ax = axes[c.SIGNAL_NAMES.index(signal_name)]
            point_sets[subject_id][signal_name]['pcs'] = []
            for (x, y), label in zip(point_sets[subject_id][signal_name]['points'], point_sets[subject_id][signal_name]['labels']):
                pc = plot_point(x, y, label, ax)
                point_sets[subject_id][signal_name]['pcs'].append(pc)
            ax.relim()
            ax.autoscale_view()

    def unplot_point(
                pc: PathCollection
            ) -> None:
        try:
            pc.remove()
        except ValueError:
            pass

    def unplot_points(
                subject_id: int
            ) -> None:
        for signal_name in c.SIGNAL_NAMES:
            for pc in point_sets[subject_id][signal_name]['pcs']:
                unplot_point(pc)

    def unplot_all_points():
        for ax in axes:
            for pc in ax.collections:
                unplot_point(pc)

    # setup variables

    point_sets: dict[int, dict[str, dict[str, list[int | PathCollection]]]] = {}

    # read samples data

    def append_record_data(
                row: pd.Series
            ) -> pd.Series:
        subject_id, segment_id, label, sample_start = row[['subject_id', 'segment_id', 'class', 'sample_start']]
        p_folder = c.RECORDS_DIR / label / f'p{subject_id:06d}'
        signals, fields = wfdb.io.rdsamp(p_folder / segment_id, channel_names=c.SIGNAL_NAMES)
        fs = fields['fs']
        s0 = sample_start + 2 * 60 * fs
        s1 = s0 + 1 * 60 * fs
        signals = signals[s0:s1, :]
        time = np.arange(signals.shape[0]) / fs
        return pd.concat([row, pd.Series({'time': time, 'signals': signals, 'fs': fs, 'sig_len': signals.shape[0]})])

    df_samples = pd.read_csv(
        c.SAMPLES_PATH,
        dtype={
            'subject_id': pd.Int64Dtype(),
            'master_id': pd.StringDtype(),
            'segment_id': pd.StringDtype(),
            'class': pd.StringDtype(),
            'sample_start': pd.Int64Dtype(),
            'sample_end': pd.Int64Dtype()
        }
    )

    df_samples = df_samples.apply(append_record_data, axis=1)

    # data functions

    def clear_points(
                subject_id: int
            ):
        for signal_name in c.SIGNAL_NAMES:
            point_sets[subject_id][signal_name] = {
                'points': [],
                'labels': [],
                'pcs': []
            }

    def save_points():
        records = []
        for subject_id, subject_dict in point_sets.items():
            for signal_name, signal_dict in subject_dict.items():
                for (x, y), label in zip(signal_dict['points'], signal_dict['labels']):
                    records.append((subject_id, signal_name, x, y, label))
        df_records = pd.DataFrame(records, columns=['subject_id', 'signal_name', 'x', 'y', 'label'])
        df_records = df_records.set_index(['subject_id'])
        df_records.to_csv(c.POINTS_PATH)

    def load_points(
                subject_id: int
            ):
        df_records = pd.read_csv(
            c.POINTS_PATH,
            dtype={
                'subject_id': pd.Int64Dtype(),
                'signal_name': pd.StringDtype(),
                'x': pd.Float64Dtype(),
                'y': pd.Float64Dtype(),
                'label': pd.Int64Dtype(),
            }
        )
        if not df_records.empty:
            df_records = df_records.set_index(['subject_id', 'signal_name'])
            for signal_name in c.SIGNAL_NAMES:
                if (subject_id, signal_name) in df_records.index:
                    point_sets[subject_id][signal_name] = {
                        'points': [],
                        'labels': [],
                        'pcs': []
                    }
                    if (subject_id, signal_name) in df_records.index:
                        for x, y, label in df_records.loc[(subject_id, signal_name), :].itertuples(index=False):
                            point_sets[subject_id][signal_name]['points'].append((x, y))
                            point_sets[subject_id][signal_name]['labels'].append(label)

    def load_all_points():
        df_records = pd.read_csv(
            c.POINTS_PATH,
            dtype={
                'subject_id': pd.Int64Dtype(),
                'signal_name': pd.StringDtype(),
                'x': pd.Float64Dtype(),
                'y': pd.Float64Dtype(),
                'label': pd.Int64Dtype(),
            }
        )
        init_all_points()
        if not df_records.empty:
            df_records = df_records.set_index(['subject_id', 'signal_name'])
            for subject_id in df_samples['subject_id']:
                for signal_name in c.SIGNAL_NAMES:
                    if (subject_id, signal_name) in df_records.index:
                        for x, y, label in df_records.loc[(subject_id, signal_name), :].itertuples(index=False):
                            point_sets[subject_id][signal_name]['points'].append((x, y))
                            point_sets[subject_id][signal_name]['labels'].append(label)

    def init_all_points():
        for subject_id in df_samples['subject_id']:
            point_sets[subject_id] = {}
            for signal_name in c.SIGNAL_NAMES:
                point_sets[subject_id][signal_name] = {
                    'points': [],
                    'labels': [],
                    'pcs': []
                }

    # load or init points

    if not c.POINTS_PATH.exists():
        init_all_points()
        LOGGER.info('file (%s) not found, running with no data', c.POINTS_PATH)
    else:
        load_all_points()
        LOGGER.info('loaded data from file (%s)', c.POINTS_PATH)

    # setup the curves

    def get_slice(subject_idx, sample_idx):
        time, signals, fs = df_samples.loc[subject_idx, ['time', 'signals', 'fs']]
        sa = int(sample_idx)
        sb = int(sample_idx) + int(np.floor(DISPLAY_TIME_WINDOW * fs))
        wt = time[sa:sb]
        ws = signals[sa:sb, :]
        return wt, ws

    wt, ws = get_slice(0, 0)

    lines: list[Line2D] = []
    for k, (ax, signal_name) in enumerate(zip(axes, c.SIGNAL_NAMES)):
        line, = ax.plot(wt, ws[:, k], color=f'C{k}', picker=True, zorder=1)
        lines.append(line)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(signal_name)

    # time slider

    ax_time: Axes = fig.add_axes([0.07, 0.00, 0.77, 0.075])
    time_slider = Slider(
        ax=ax_time,
        label='Sample [#]',
        valmin=0,
        valmax=1,
        valinit=0,
        valstep=1,
        valfmt='%d'
    )

    def update_sample(sample_idx):
        subject_idx = int(subject_slider.val)
        wt, ws = get_slice(subject_idx, sample_idx)
        for k, (line, ax) in enumerate(zip(lines, axes)):
            line: plt.Line2D
            line.set_data(wt, ws[:, k])
            ax.relim()
            ax.autoscale_view()
        fig.canvas.draw_idle()

    time_slider.on_changed(update_sample)

    # subject slider

    ax_subject: Axes = fig.add_axes([0.07, 0.05, 0.77, 0.075])
    subject_slider = Slider(
        ax=ax_subject,
        label='Subject [#]',
        valmin=0,
        valmax=len(df_samples)-1,
        valinit=0,
        valstep=1,
        valfmt='%d'
    )

    def update_subject(subject_idx):
        subject_id, label, sig_len, fs = df_samples.loc[subject_idx, ['subject_id', 'class', 'sig_len', 'fs']]
        new_max = sig_len - fs * DISPLAY_TIME_WINDOW - 1
        sample_start = 0
        time_slider.set_val(sample_start)
        time_slider.valmax = new_max
        ax_time.set_xlim([0, new_max])
        subject_slider.valtext.set_text(f'{subject_idx} ({label}) ({subject_id})')
        unplot_points(subject_id)
        unplot_all_points()
        plot_points(subject_id)
        update_sample(sample_start)

    subject_slider.on_changed(update_subject)

    # save button

    ax_save = fig.add_axes([0.95, 0.08, 0.04, 0.025])
    save_button = Button(ax_save, 'save', hovercolor='0.975')

    def save(_event):
        save_points()
        LOGGER.info('saved data to file (%s)', c.POINTS_PATH)

    save_button.on_clicked(save)

    # clear button

    ax_clear = fig.add_axes([0.95, 0.05, 0.04, 0.025])
    clear_button = Button(ax_clear, 'clear', hovercolor='0.975')

    def clear(_event):
        subject_idx = int(subject_slider.val)
        subject_id = int(df_samples.loc[subject_idx, 'subject_id'])
        unplot_points(subject_id)
        unplot_all_points()
        clear_points(subject_id)
        fig.canvas.draw_idle()
        LOGGER.info('cleared points for subject %d', subject_id)

    clear_button.on_clicked(clear)

    # load button

    ax_load = fig.add_axes([0.95, 0.02, 0.04, 0.025])
    load_button = Button(ax_load, 'load', hovercolor='0.975')

    def load(_event):
        subject_idx = int(subject_slider.val)
        subject_id = int(df_samples.loc[subject_idx, 'subject_id'])
        unplot_points(subject_id)
        unplot_all_points()
        load_points(subject_id)
        plot_points(subject_id)
        fig.canvas.draw_idle()
        LOGGER.info('reloaded points for subject %d from file (%s)', subject_id, c.POINTS_PATH)

    load_button.on_clicked(load)

    # neurokit button

    ax_nk = fig.add_axes([0.90, 0.02, 0.04, 0.025])
    nk_button = Button(ax_nk, 'nk', hovercolor='0.975')

    def make_filter(design: str, wn: int, fs: float):
        if design == 'cheby2':
            filt = scipy.signal.cheby2(4, 20, wn, btype='band' if isinstance(wn, list) else 'low', fs=fs, output='sos')
        elif design == 'butter':
            filt = scipy.signal.butter(4, wn, btype='band' if isinstance(wn, list) else 'low', fs=fs, output='sos')
        elif design == 'bessel':
            filt = scipy.signal.bessel(4, wn, btype='band' if isinstance(wn, list) else 'low', fs=fs, output='sos')
        else:
            raise NotImplementedError
        return filt

    def detect_peaks_nk(
                subject_idx: int,
                subject_id: int
            ):
        time, signals, fs = df_samples.loc[subject_idx, ['time', 'signals', 'fs']]
        ppg = signals[:, c.SIGNAL_NAMES.index('PLETH')]
        design = 'butter'
        wn = [0.5, 30]
        filt = make_filter(design, wn, fs)
        ppg_filt = scipy.signal.sosfiltfilt(filt, ppg)
        _, ppg_info = nk.ppg.ppg_peaks(ppg_cleaned=ppg_filt, sampling_rate=fs, method='charlton')
        peaks = ppg_info['PPG_Peaks']
        peak_points = np.stack((time[peaks], ppg[peaks]), axis=1)
        onsets = ppg_info['PPG_Onsets']
        onset_points = np.stack((time[onsets], ppg[onsets]), axis=1)
        points = np.concat((peak_points, onset_points)).tolist()
        labels = np.concat((np.ones_like(peaks) * 1, np.ones_like(onsets) * 3)).astype(int).tolist()
        point_sets[subject_id]['PLETH'] = {
            'points': points,
            'labels': labels,
            'pcs': []
        }

    def run_nk(_event):
        subject_idx = int(subject_slider.val)
        subject_id = int(df_samples.loc[subject_idx, 'subject_id'])
        unplot_points(subject_id)
        unplot_all_points()
        detect_peaks_nk(subject_idx, subject_id)
        plot_points(subject_id)
        fig.canvas.draw_idle()
        LOGGER.info('ran peak detection with neurokit2 for subject %d', subject_id)

    nk_button.on_clicked(run_nk)

    # click callback

    def on_pick(event: PickEvent):

        button = event.mouseevent.button

        if button in [MouseButton.LEFT, MouseButton.RIGHT, MouseButton.MIDDLE]:
            LOGGER.debug('================ on pick called ================')

            new_label = button

            subject_idx = int(subject_slider.val)
            subject_id = int(df_samples.loc[subject_idx, 'subject_id'])
            ax = event.mouseevent.inaxes
            signal_name = ax.get_ylabel()
            points = point_sets[subject_id][signal_name]['points']
            labels = point_sets[subject_id][signal_name]['labels']
            pcs = point_sets[subject_id][signal_name]['pcs']

            LOGGER.debug(ax.collections)

            if isinstance(event.artist, Line2D):
                LOGGER.debug('\tartist is a line')
                is_also_point = any(p.contains(event.mouseevent)[0] for p in pcs)
                if not is_also_point:
                    LOGGER.debug('\t\tbut not a point, adding')
                    xdata = event.artist.get_xdata()
                    ydata = event.artist.get_ydata()
                    picked_point_indices = event.ind
                    picked_point_index = picked_point_indices[len(picked_point_indices) // 2]
                    x_point = xdata[picked_point_index]
                    y_point = ydata[picked_point_index]
                    pc = plot_point(x_point, y_point, new_label, ax)
                    points.append((x_point, y_point))
                    labels.append(new_label)
                    pcs.append(pc)
                else:
                    LOGGER.debug('\t\tbut also a point, skipping')

            else:
                LOGGER.debug('\tartist is a point')
                picked_point_index = pcs.index(event.artist)
                if new_label == labels[picked_point_index]:
                    LOGGER.debug('\t\tpoint is same type (%s), removing point', new_label)
                    event.artist.remove()
                    points.pop(picked_point_index)
                    labels.pop(picked_point_index)
                    pcs.pop(picked_point_index)
                else:
                    LOGGER.debug('\t\tpoint is diffent type, switching old type (%s) to new type (%s)', labels[picked_point_index], new_label)
                    labels[picked_point_index] = new_label
                    pcs[picked_point_index].set_color(DISPLAY_POINT_COLOR_MAP[new_label])

            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)

    # scroll callback

    def on_scroll(event: MouseEvent):
        step = (-event.step) * df_samples.loc[int(subject_slider.val), 'fs'] // 5
        new_time_val = np.clip(time_slider.val + step, 0, time_slider.valmax, dtype=int)
        time_slider.set_val(new_time_val)

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # start with first subject

    update_subject(0)

    plt.show()


if __name__ == '__main__':
    main()
