from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
from matplotlib import pyplot as plt
from tqdm import tqdm


def main():

    csvs_dir = Path('../data/csvs')

    ppg_dir = Path('../data/ppg')

    images_dir = Path('../images')

    df_subjects = pd.read_csv(csvs_dir / 'subjects.csv')

    # patient_files = sorted(ppg_dir.iterdir())

    df_subjects['patient_file'] = df_subjects.apply(lambda r: ppg_dir / f'p{r["subject_id"]:06d}_{r["segment_id"]}.mat', axis=1)

    # df_samples = df_subjects.groupby('class').sample(5, random_state=0)[['subject_id', 'class', 'patient_file']].sort_values(['class', 'subject_id'])
    df_samples = df_subjects[['subject_id', 'class', 'patient_file']].sort_values(['class', 'subject_id'])

    fs = 125

    experiment_sets = {

        # title
        #   filter      order   freqs   att     label

        # 'frequency search: lowpass, butterworth, 4th order': [
        #     (None,      None,   None,   None,   'raw'),
        #     ('butter',  4,      10,     None,   'butter, 10 hz'),
        #     ('butter',  4,      30,     None,   'butter, 30 hz'),
        # ],

        # 'attenuation search: chebyshev type 2, 4th order, lowpass, 10 hz': [
        #     (None,      None,   None,   None,   'raw'),
        #     ('cheby2',  4,      10,     20,     'cheby2, 10 hz, 20 db'),
        #     ('cheby2',  4,      10,     40,     'cheby2, 10 hz, 40 db'),
        #     ('cheby2',  4,      10,     60,     'cheby2, 10 hz, 60 db'),
        #     ('cheby2',  4,      10,     80,     'cheby2, 10 hz, 80 db'),
        # ],

        # 'attenuation search: chebyshev type 2, 4th order, lowpass, 30 hz': [
        #     (None,      None,   None,   None,   'raw'),
        #     ('cheby2',  4,      30,     20,     'cheby2, 30 hz, 20 db'),
        #     ('cheby2',  4,      30,     40,     'cheby2, 30 hz, 40 db'),
        #     ('cheby2',  4,      30,     60,     'cheby2, 30 hz, 60 db'),
        #     ('cheby2',  4,      30,     80,     'cheby2, 30 hz, 80 db'),
        # ],

        # 'frequency search: chebyshev type 2, 4th order, lowpass, 20 db': [
        #     (None,      None,   None,   None,   'raw'),
        #     ('cheby2',  4,      10,     20,     'cheby2, 10 hz, 20 db'),
        #     ('cheby2',  4,      15,     20,     'cheby2, 15 hz, 20 db'),
        #     ('cheby2',  4,      25,     20,     'cheby2, 25 hz, 20 db'),
        #     ('cheby2',  4,      30,     20,     'cheby2, 30 hz, 20 db'),
        #     ('cheby2',  4,      35,     20,     'cheby2, 35 hz, 20 db'),
        # ],

        # 'frequency search: chebyshev type 2, 4th order, bandpass, 20 db': [
        #     ('detrend', None,   None,           None,   'detrend'),
        #     ('cheby2d', 4,      30,             20,     'cheby2d, 30 hz, 20 db'),
        #     ('cheby2',  4,      [0.05, 30],     20,     'cheby2, 0.05 - 30 hz, 20 db'),
        #     ('cheby2',  4,      [0.1, 30],      20,     'cheby2, 0.1 - 30 hz, 20 db'),
        #     ('cheby2',  4,      [0.5, 30],      20,     'cheby2, 0.5 - 30 hz, 20 db'),
        # ],

        # 'frequency search: lowpass, butterworth, 4th order': [
        #     (None,      None,   None,   None,   'raw'),
        #     ('butter',  4,      10,     None,   'butter, 10 hz'),
        #     ('butter',  4,      30,     None,   'butter, 30 hz'),
        #     ('cheby2',  4,      10,     20,     'cheby2, 10 hz, 20 db'),
        #     ('cheby2',  4,      30,     20,     'cheby2, 30 hz, 20 db'),
        # ],

        'frequency search: chebyshev type 2, 4th order, 20 db': [
            (None,      None,   None,   None,   'raw'),
            ('cheby2',  4,      10,     20,     'cheby2, 10 hz, 20 db'),
            ('cheby2',  4,      15,     20,     'cheby2, 15 hz, 20 db'),
            ('cheby2',  4,      25,     20,     'cheby2, 25 hz, 20 db'),
            ('cheby2',  4,      30,     20,     'cheby2, 30 hz, 20 db'),
            ('cheby2',  4,      35,     20,     'cheby2, 35 hz, 20 db'),
        ],

        'attenuation search: chebyshev type 2, 4th order, 30 hz': [
            (None,      None,   None,   None,   'raw'),
            ('cheby2',  4,      30,     20,     'cheby2, 30 hz, 20 db'),
            ('cheby2',  4,      30,     40,     'cheby2, 30 hz, 40 db'),
            ('cheby2',  4,      30,     60,     'cheby2, 30 hz, 60 db'),
            ('cheby2',  4,      30,     80,     'cheby2, 30 hz, 80 db'),
        ],

        'frequency search: butterworth, 4th order': [
            (None,      None,   None,   None,   'raw'),
            ('butter',  4,      10,     None,   'butter, 10 hz'),
            ('butter',  4,      15,     None,   'butter, 15 hz'),
            ('butter',  4,      25,     None,   'butter, 25 hz'),
            ('butter',  4,      30,     None,   'butter, 30 hz'),
            ('butter',  4,      35,     None,   'butter, 35 hz'),
        ],

        'comparison: butterworth, 4th order, 30 hz vs chebyshev type 2, 4th order, 30 hz, 20 db': [
            (None,      None,   None,       None,   'raw'),
            ('butter',  4,      30,         None,   'butter, 30 hz'),
            ('butter',  4,      [0.5, 30],  None,   'butter, 0.5 - 30 hz'),
            ('cheby2',  4,      30,         20,     'cheby2, 30 hz, 20 db'),
            ('cheby2',  4,      [0.5, 30],  20,     'cheby2, 0.5 - 30 hz, 20 db'),
        ],

    }

    alpha = 1.0

    plt.rcParams.update({'font.size': 20})

    iterator = list(df_samples.itertuples(index=False)) # [-1:] # [-1:]

    for subject_id, category, patient_file in tqdm(iterator, position=0):

        if subject_id not in [64485, 54586]:
            continue

        mat = scipy.io.loadmat(patient_file)

        t = mat['time'].squeeze()
        ppg = mat['data'].squeeze()

        for name, experiments in tqdm(experiment_sets.items(), position=1, leave=False):

            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(32, 16))
            axes: tuple[plt.Axes]

            for filter_type, n, wn, rs, label in experiments:
                kwargs = {'alpha': alpha, 'label': label}
                if filter_type is None:
                    kwargs['color'] = 'black'
                    ppgf = ppg
                elif filter_type == 'detrend':
                    kwargs['color'] = 'black'
                    ppgf = scipy.signal.detrend(ppg, type='constant')
                elif filter_type == 'butter':
                    b, a = scipy.signal.butter(n, wn, btype='band' if isinstance(wn, list) else 'low', fs=fs)
                    ppgf = scipy.signal.filtfilt(b, a, ppg)
                elif filter_type == 'cheby2':
                    b, a = scipy.signal.cheby2(n, rs, wn, btype='band' if isinstance(wn, list) else 'low', fs=fs)
                    ppgf = scipy.signal.filtfilt(b, a, ppg)
                elif filter_type == 'cheby2d':
                    ppgf = scipy.signal.detrend(ppg, type='constant')
                    b, a = scipy.signal.cheby2(n, rs, wn, btype='band' if isinstance(wn, list) else 'low', fs=fs)
                    ppgf = scipy.signal.filtfilt(b, a, ppgf)
                else:
                    raise NotImplementedError

                axes[0].plot(t, ppgf, **kwargs)
                axes[1].plot(t, np.gradient(ppgf, t), **kwargs)
                axes[2].plot(t, np.gradient(np.gradient(ppgf, t), t), **kwargs)

            axes[0].set_ylabel('PPG')
            axes[0].legend()
            axes[0].grid(True)

            axes[1].set_ylabel('VPG')
            # axes[1].legend()
            axes[1].grid(True)

            axes[2].set_xlabel('time')
            axes[2].set_ylabel('APG')
            # axes[2].legend()
            axes[2].grid(True)

            fig.suptitle(name)

            plt.xlim(149, 151)
            # plt.xlim(147, 153)
            # plt.xlim(0, 6)

            # lims = [149, 151]
            # i = np.where((t > lims[0]) & (t < lims[1]))[0]
            # vpgf = np.gradient(ppgf, t)
            # apgf = np.gradient(np.gradient(ppgf, t), t)
            # axes[0].set_ylim(ppgf[i].min()-0.01, ppgf[i].max()+0.01)
            # axes[1].set_ylim(vpgf[i].min()-0.01, vpgf[i].max()+0.01)
            # axes[2].set_ylim(apgf[i].min()-0.01, apgf[i].max()+0.01)

            fig.tight_layout()
            plt.show()

            subject_dir = images_dir / f'{category}_{subject_id}'
            subject_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(subject_dir / name)

            plt.close()


if __name__ == '__main__':
    main()
