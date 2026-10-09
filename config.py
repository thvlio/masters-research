from pathlib import Path


class Config:

    # paths

    DATA_DIR = Path('data')

    IMAGES_DIR = Path('images')

    CRSIDLAB_DIR = Path('crsidlab') / 'CRSIDLab_Sept2024'

    CSV_DIR = DATA_DIR / 'csvs'

    RECORDS_DIR = DATA_DIR / 'selected'

    PPG_DIR = DATA_DIR / 'ppg'

    PYPPG_DIR = DATA_DIR / 'pyppg'

    PYPPG_TEMP_DIR = PYPPG_DIR / 'temp'

    SUBJECTS_PATH = CSV_DIR / 'subjects.csv'

    SAMPLES_PATH = CSV_DIR / 'samples.csv'

    POINTS_PATH = CSV_DIR / 'points.csv'

    # signals

    SIGNAL_NAMES = ['II', 'PLETH', 'ABP']

    SIGNAL_FS = 125

    # processing

    NUM_SELECTED_CYCLES = 30

    NUM_DISCARDED_EDGE_CYCLES = 2
