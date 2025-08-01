import gzip
import shutil
from pathlib import Path

from tqdm import tqdm


def main():
    for fp in tqdm(sorted(Path('/home/thulio/Downloads/physionet.org/files/mimiciii/1.4').iterdir())):
        if fp.name.endswith('.csv.gz'):
            with gzip.open(fp, 'rb') as f_gz:
                new_fp = f'{fp.name.split('.')[0]}.csv'
                with open(new_fp, 'wb') as f_csv:
                    shutil.copyfileobj(f_gz, f_csv)


if __name__ == '__main__':
    main()
