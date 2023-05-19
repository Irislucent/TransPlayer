import os
from glob import glob

import numpy as np
import librosa as lr
from soundfile import write

CROP_DURATION = 200 # seconds

def crop_full(filepath, out_dir):
    ins_name = os.path.basename(filepath).split("_")[0]
    y, sr = lr.load(filepath, sr=16000)
    n_samples_chunk = sr * CROP_DURATION
    chunk_idx = 0
    while chunk_idx * n_samples_chunk < len(y):
        chunk = y[chunk_idx * n_samples_chunk : min((chunk_idx + 1) * n_samples_chunk, len(y))]
        chunk_name = ins_name + '_' + str(chunk_idx).zfill(2) + '.wav' 
        write(
            os.path.join(out_dir, chunk_name),
            chunk,
            sr
        )
        chunk_idx += 1
        print("Wrote", chunk_name)


if __name__ == "__main__":
    full_data_dir = "../../../data/full"
    out_dir = "../../../data"
    filepaths = glob(f"{full_data_dir}/*.wav", recursive=True)
    for filepath in filepaths:
        crop_full(filepath, out_dir)