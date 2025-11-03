from pathlib import Path

import numpy as np
import pandas as pd
# import scipy.io
import wfdb.io
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider # Button


def main():

    dataset_root = Path('/home/thulio/projects/masters-research/data')

    signal_names = ['II', 'PLETH', 'ABP']

    window = 5 * 60

    df_subjects = pd.read_csv('csvs/subjects.csv', index_col=0)

    subject_data = []
    for subject_id, _, segment_id, _, label in df_subjects.itertuples(index=False):
        p_folder = dataset_root / label / f'p{subject_id:06d}'
        signals, fields = wfdb.io.rdsamp(p_folder / segment_id, channel_names=signal_names)
        time = np.arange(fields['sig_len']) / fields['fs']
        subject_data.append((subject_id, time, signals, fields))

    def get_slice(subject_idx, sample_idx):
        _, time, signals, fields = subject_data[subject_idx]
        sa = int(sample_idx)
        sb = int(sample_idx) + int(np.floor(window * fields['fs']))
        wt = time[sa:sb]
        ws = signals[sa:sb, :]
        return wt, ws

    fig, axes = plt.subplots(3, 1, figsize=(16, 8))
    axes: tuple[plt.Axes]
    plt.tight_layout()
    wt, ws = get_slice(0, 0)
    lines = []
    for k, (ax, signal_name) in enumerate(zip(axes, signal_names)):
        line, = ax.plot(wt, ws[:, k])
        lines.append(line)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(signal_name)
    fig.subplots_adjust(bottom=0.25)

    ax_time: plt.Axes = fig.add_axes([0.15, 0.04, 0.7, 0.1])
    time_slider = Slider(
        ax=ax_time,
        label='Sample [#]',
        valmin=0,
        valmax=subject_data[0][-1]['sig_len'] - subject_data[0][-1]['fs'] * window - 1,
        valinit=0,
        valstep=1,
        valfmt='%d'
    )

    ax_subject: plt.Axes = fig.add_axes([0.15, 0.12, 0.7, 0.1])
    subject_slider = Slider(
        ax=ax_subject,
        label='Subject [#]',
        valmin=0,
        valmax=len(df_subjects)-1,
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

    def update_subject(subject_idx):
        new_max = subject_data[subject_idx][-1]['sig_len'] - subject_data[subject_idx][-1]['fs'] * window - 1
        time_slider.reset()
        time_slider.valmax = new_max
        ax_time.set_xlim([0, new_max])
        update_sample(0)

    subject_slider.on_changed(update_subject)

    # resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    # button = Button(resetax, 'Reset', hovercolor='0.975')

    # def reset(event):
    #     time_slider.reset()

    # button.on_clicked(reset)

    plt.show()


if __name__ == '__main__':
    main()
