import os
from pathlib import Path

dir = 'src'
files_name=[
    f'{dir}/models/encoder.py',
    f'{dir}/models/decoder.py',
    f'{dir}/laguage/LangSegment.py',
    f'{dir}/laguage/language_filters.py',

    f'{dir}/preprocessing/data_loader.py',
    f'{dir}/preprocessing/preprocess.py'

]

for files in files_name:
    path_file = Path(files)

    floder ,file  = os.path.split(path_file)
    
    if not os.path.exists(floder):
        os.makedirs(floder)

    if not os.path.exists(path_file):
        with open(path_file, 'w') as f:
            pass
    