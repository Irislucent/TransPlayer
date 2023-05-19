import os
import time
import datetime

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from matplotlib import pyplot as plt

from model import Generator


class Trainer(object):
    def __init__(self, dataloader, config):
        """Initialize configurations."""

        # Data loader.
        self.dataloader = dataloader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.lambda_cd_cross = config.lambda_cd_cross
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters

        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.log_step = config.log_step
        self.save_step = config.save_step
        self.save_dir = config.save_dir
        self.summary_writer = None

        # Build the model and tensorboard.
        self.build_model()

    def build_model(self):

        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 1e-4)

        self.G.to(self.device)

        # load previous model
        cp_path = os.path.join(
            self.save_dir, "weights_log_cqt_down32_neck32_onehot4_withcross"
        )
        if os.path.exists(cp_path):
            save_info = torch.load(cp_path)
            self.G.load_state_dict(save_info["model"])
            self.g_optimizer.load_state_dict(save_info["optimizer"])
            self.g_optimizer.state_dict()["param_groups"][0]["lr"] /= 2

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()

    def train(self):
        # Set data loader.
        dataloader = dataloader

        # Print logs in specified order
        keys = ["G/loss_id", "G/loss_id_psnt", "G/loss_cd", "G/loss_cd_cross"]

        # Start training.
        print("Start training...")
        start_time = time.time()
        self.G = self.G.train()

        for i in range(self.num_iters):
            # Fetch data.
            try:
                x_real, repr_org, repr_trg = next(data_iter)
            except:
                data_iter = iter(dataloader)
                x_real, repr_org, repr_trg = next(data_iter)

            x_real = x_real.to(self.device)
            repr_org = repr_org.to(self.device)
            repr_trg = repr_trg.to(self.device)

            # Reconstruction loss
            x_identic, x_identic_psnt, c_x_identic = self.G(x_real, repr_org, repr_org)
            x_identic = torch.squeeze(x_identic, dim=1)
            x_identic_psnt = torch.squeeze(x_identic_psnt, dim=1)
            g_loss_id = F.l1_loss(x_real, x_identic)
            g_loss_id_psnt = F.l1_loss(x_real, x_identic_psnt)

            # Content consistency loss
            c_x_identic_recon = self.G(x_identic_psnt, repr_org, None)
            g_loss_cd = F.l1_loss(c_x_identic, c_x_identic_recon)

            # cross-domain content consistency loss
            x_trans, x_trans_psnt, c_x_trans = self.G(x_real, repr_org, repr_trg)
            c_x_trans_recon = self.G(x_trans_psnt, repr_trg, None)
            g_loss_cd_cross = F.l1_loss(c_x_trans, c_x_trans_recon)

            # Backward and optimize.
            g_loss = (
                g_loss_id
                + g_loss_id_psnt
                + self.lambda_cd * g_loss_cd
                + self.lambda_cd_cross * g_loss_cd_cross
            )
            self.reset_grad()

            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss["G/loss_id"] = g_loss_id.item()
            loss["G/loss_id_psnt"] = g_loss_id_psnt.item()
            loss["G/loss_cd"] = g_loss_cd.item()
            loss["G/loss_cd_cross"] = g_loss_cd_cross.item()

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(
                    et, i + 1, self.num_iters
                )
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)
                self._write_summary(
                    i,
                    g_loss,
                    loss["G/loss_id"],
                    loss["G/loss_id_psnt"],
                    loss["G/loss_cd"],
                    loss["G/loss_cd_cross"],
                )

            # save model checkpoint
            if i % self.save_step == 1 and i > 1000:
                save_info = {
                    "iteration": i,
                    "model": self.G.state_dict(),
                    "optimizer": self.g_optimizer.state_dict(),
                }
                save_name = "cp.pt"
                save_path = os.path.join(self.save_dir, save_name)
                torch.save(save_info, save_path)

    def _write_summary(self, i, loss, loss_id, loss_id_psnt, loss_cd, loss_cd_cross):
        writer = self.summary_writer or SummaryWriter(self.save_dir, purge_step=i)
        writer.add_scalar("train/loss_all", loss, i)
        writer.add_scalar("train/loss_id", loss_id, i)
        writer.add_scalar("train/loss_id_psnt", loss_id_psnt, i)
        writer.add_scalar("train/loss_cd", loss_cd, i)
        writer.add_scalar("train/loss_cd_cross", loss_cd_cross, i)
        writer.flush()

        self.summary_writer = writer
