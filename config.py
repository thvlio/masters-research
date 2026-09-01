from pathlib import Path


class Config:

    DATA_DIR = Path('data')

    CSV_DIR = DATA_DIR / 'csvs'

    RECORDS_DIR = DATA_DIR / 'selected'

    PPG_DIR = DATA_DIR / 'ppg'

    PYPPG_DIR = DATA_DIR / 'pyppg'

    PYPPG_TEMP_DIR = DATA_DIR = PYPPG_DIR / 'temp'

    SUBJECTS_PATH = CSV_DIR / 'subjects.csv'

    SAMPLES_PATH = CSV_DIR / 'samples.csv'

    POINTS_PATH = CSV_DIR / 'points.csv'

    SIGNAL_NAMES = ['II', 'PLETH', 'ABP']
