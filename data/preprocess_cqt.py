import numpy as np
import librosa as lr
import os

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm

OUT_DIR = "../../../data"

def transform(filepath):
    filename = os.path.basename(filepath)
    audio, sr = lr.load(filepath, sr=None)
    assert sr == 16000
    cqt_representation = lr.cqt(audio, sr=sr, hop_length=256)

    cqt_magnitude = np.abs(cqt_representation)

    print(cqt_representation.shape)

    print(os.path.join(OUT_DIR, f"{filename}_cqt.npy"))
    np.save(os.path.join(OUT_DIR, f"{filename}_cqt.npy"), cqt_magnitude)


def main(args):
    filepaths = glob(f"{args.dir}/*.wav", recursive=True)
    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(transform, filepaths),
                desc="Preprocessing",
                total=len(filepaths),
            )
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        default="../../../data_syn/cropped",
        help="directory containing .wav files",
    )
    main(parser.parse_args())
