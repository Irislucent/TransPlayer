import os
import argparse
from trainer import Trainer
from dataloader import get_dataloader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ("true")


def main(config):
    # For fast training.
    cudnn.benchmark = True

    dataloader = get_dataloader(config.data_dir, config.batch_size, config.len_crop)

    solver = Trainer(dataloader, config)

    solver.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument(
        "--lambda_cd", type=float, default=1, help="weight for hidden code loss"
    )
    parser.add_argument(
        "--lambda_cd_cross", type=float, default=0, help="weight for hidden code loss"
    )
    parser.add_argument("--dim_neck", type=int, default=32)
    parser.add_argument("--dim_emb", type=int, default=4)  # one-hot 8 to 4
    parser.add_argument("--dim_pre", type=int, default=512)
    parser.add_argument("--freq", type=int, default=32)

    # Training configuration.
    parser.add_argument("--data_dir", type=str, default="../../../data_syn/cropped")
    parser.add_argument("--batch_size", type=int, default=2, help="mini-batch size")
    parser.add_argument(
        "--num_iters", type=int, default=1000000, help="number of total iterations"
    )
    parser.add_argument(
        "--len_crop", type=int, default=128, help="dataloader output sequence length"
    )

    # Miscellaneous.
    parser.add_argument("--log_step", type=int, default=25)
    parser.add_argument("--save_step", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="../../../autoencoder_cp")

    config = parser.parse_args()
    print(config)
    main(config)
