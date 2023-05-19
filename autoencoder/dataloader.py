import os
import random

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

INS_LIST = ["epiano", "flute", "guitar", "harp", "piano", "organ", "trumpet", "viola"]

class InstrumentChunkDataset(Dataset):
    def __init__(self, dataset_dir, len_crop):
        self.dataset_dir = dataset_dir
        self.len_crop = len_crop
        self.eps = 1e-8

        loaded_data = {}
        # the first terms are the one-hot vectors
        for ins_index, ins_name in enumerate(INS_LIST):
            # one-hot
            loaded_data[ins_name] = [ins_index]
        # the rest are the input representations
        for filename in os.listdir(self.dataset_dir):
            if os.path.splitext(filename)[1] == ".npy":
                filepath = os.path.join(self.dataset_dir, filename)
                ins_name = filename.split("_")[0]
                loaded_data[ins_name].append(
                    np.log(np.load(filepath) + self.eps)
                )  # log cqt

        self.loaded_data = loaded_data

        self.num_tokens = len(list(self.loaded_data.keys()))

    def __getitem__(self, index):
        loaded_data = self.loaded_data

        # pick a random original instrument
        list_ins = list(loaded_data.keys())
        list_chunks = loaded_data[list_ins[index]]
        repr_org = list_chunks[0]

        # pick random chunk and apply random crop
        random_idx = np.random.randint(1, len(list_chunks))
        chunk = list_chunks[random_idx].T
        if chunk.shape[0] < self.len_crop:
            len_pad = self.len_crop - chunk.shape[0]
            sample = np.pad(chunk, ((0, len_pad), (0, 0)), "constant")
        elif chunk.shape[0] > self.len_crop:
            left = np.random.randint(chunk.shape[0] - self.len_crop)
            sample = chunk[left : left + self.len_crop, :]
        else:
            sample = chunk

        # pick random target
        repr_trg = loaded_data[random.choice(list_ins)][0]

        return sample, repr_org, repr_trg

    def __len__(self):
        return self.num_tokens


def get_dataloader(dataset_dir, batch_size=16, len_crop=128, num_workers=0):
    dataset = InstrumentChunkDataset(dataset_dir, len_crop)

    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    return dataloader


if __name__ == "__main__":
    dataset_dir = "../../../data_syn/cropped"
    batch_size = 2
    len_crop = 128
    num_workers = 0

    dataloader = get_dataloader(dataset_dir, batch_size, len_crop, num_workers)
    for i, (x, y_org, y_trg) in enumerate(dataloader):
        print(x.shape)
        print(y_org.shape)
        print(y_trg.shape)
        break
