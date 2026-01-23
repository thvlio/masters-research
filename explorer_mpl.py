from pathlib import Path

import numpy as np
import pandas as pd
import wfdb.io
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button


def main():

    plt.rcParams.update({'font.size': 18})

    dataset_root = Path('/home/thulio/projects/masters-research/data')

    csvs_root = Path('/home/thulio/projects/masters-research/csvs')

    signal_names = ['II', 'PLETH', 'ABP']

    window = 5 * 60

    df_subjects = pd.read_csv(csvs_root / 'subjects.csv')

    if (csvs_root / 'samples.csv').exists():
        df_samples_prev = pd.read_csv(csvs_root / 'samples.csv', dtype={'sample_start': 'Int64'})
        df_samples = df_subjects.merge(df_samples_prev, on=['subject_id', 'master_id', 'segment_id'], how='left')
    else:
        df_samples = df_subjects.copy()
        df_samples['sample_start'] = pd.NA

    def append_subject_data(row: pd.Series):
        subject_id, segment_id, label = row[['subject_id', 'segment_id', 'class']]
        p_folder = dataset_root / label / f'p{subject_id:06d}'
        signals, fields = wfdb.io.rdsamp(p_folder / segment_id, channel_names=signal_names)
        time = np.arange(fields['sig_len']) / fields['fs']
        return pd.concat([row, pd.Series({'time': time, 'signals': signals}), pd.Series({f'fields.{k}': v for k, v in fields.items()})])

    df_samples = df_samples.apply(append_subject_data, axis=1)

    def get_slice(subject_idx, sample_idx):
        time, signals, fs = df_samples.loc[subject_idx, ['time', 'signals', 'fields.fs']]
        sa = int(sample_idx)
        sb = int(sample_idx) + int(np.floor(window * fs))
        wt = time[sa:sb]
        ws = signals[sa:sb, :]
        return wt, ws

    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True, layout='constrained')
    axes: tuple[plt.Axes]
    # plt.tight_layout()
    wt, ws = get_slice(0, 0)
    lines = []
    for k, (ax, signal_name) in enumerate(zip(axes, signal_names)):
        line, = ax.plot(wt, ws[:, k])
        lines.append(line)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(signal_name)
    # fig.get_layout_engine().set(h_pad=0.2, w_pad=0.2, rect=(0, 0.25, 1, 0.75))
    fig.get_layout_engine().set(rect=(0.005, 0.12, 0.990, 0.875))
    # fig.subplots_adjust(bottom=0.25)

    ax_time: plt.Axes = fig.add_axes([0.07, 0.00, 0.77, 0.075])
    time_slider = Slider(
        ax=ax_time,
        label='Sample [#]',
        valmin=0,
        valmax=df_samples.loc[0, 'fields.sig_len'] - df_samples.loc[0, 'fields.fs'] * window - 1,
        valinit=0,
        valstep=1,
        valfmt='%d'
    )

    ax_subject: plt.Axes = fig.add_axes([0.07, 0.05, 0.77, 0.075])
    subject_slider = Slider(
        ax=ax_subject,
        label='Subject [#]',
        valmin=0,
        valmax=len(df_samples)-1,
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
        subject_id, label, sig_len, fs = df_samples.loc[subject_idx, ['subject_id', 'class', 'fields.sig_len', 'fields.fs']]
        new_max = sig_len - fs * window - 1
        sample_start = df_samples.loc[subject_idx, 'sample_start']
        update_stored(sample_start)
        sample_start = int(sample_start) if pd.notna(sample_start) else 0
        time_slider.set_val(sample_start)
        time_slider.valmax = new_max
        ax_time.set_xlim([0, new_max])
        subject_slider.valtext.set_text(f'{subject_idx} ({label}) ({subject_id})')
        update_sample(sample_start)

    subject_slider.on_changed(update_subject)

    stored_text = fig.text(0.94, 0.12, 'test')

    def update_stored(sample_idx):
        stored_text.set_text(f'stored: {sample_idx}')
        fig.canvas.draw_idle()

    ax_save = fig.add_axes([0.94, 0.08, 0.04, 0.025])
    save_button = Button(ax_save, 'store', hovercolor='0.975')

    ax_reset = fig.add_axes([0.94, 0.05, 0.04, 0.025])
    reset_button = Button(ax_reset, 'clear', hovercolor='0.975')

    ax_load = fig.add_axes([0.94, 0.02, 0.04, 0.025])
    load_button = Button(ax_load, 'restore', hovercolor='0.975')

    def save(_event):
        df_samples.loc[int(subject_slider.val), 'sample_start'] = time_slider.val
        df_samples.to_csv(csvs_root / 'samples.csv', index=False, columns=['subject_id', 'master_id', 'segment_id', 'sample_start'])
        df_samples.to_csv(csvs_root / 'samples_full.csv', index=False)
        update_stored(int(time_slider.val))

    save_button.on_clicked(save)

    def reset(_event):
        df_samples.loc[int(subject_slider.val), 'sample_start'] = pd.NA
        df_samples.to_csv(csvs_root / 'samples.csv', index=False, columns=['subject_id', 'master_id', 'segment_id', 'sample_start'])
        df_samples.to_csv(csvs_root / 'samples_full.csv', index=False)
        update_stored(pd.NA)

    reset_button.on_clicked(reset)

    def load(_event):
        sample_start = df_samples.loc[int(subject_slider.val), 'sample_start']
        update_stored(sample_start)
        sample_start = int(sample_start) if pd.notna(sample_start) else 0
        time_slider.set_val(sample_start)

    load_button.on_clicked(load)

    update_subject(0)

    plt.show()


if __name__ == '__main__':
    main()
