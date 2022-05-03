import time
import optuna
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import os
from datetime import datetime
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track

# Torch
import torch
from torch.utils.data import DataLoader, Dataset

# Custom trainer
import torchsuite
from torchsuite.Trainer import Trainer
from torchsuite.TrainingBoard import TrainingBoard
import pytorchcheckpoint

# Project
from deepgesture.Dataset.BlobDataset import gestureBlobMultiDataset, size_collate_fn
from deepgesture.Dataset.UnsuperviseBlobDataset import UnsupervisedBlobDatasetProbabilistic
from deepgesture.Models.EncoderDecoder import encoderDecoder
from deepgesture.Models.OpticalFlowKinematicEncoder import OKNetV1
from deepgesture.config import Config

from torchsuite.utils.Logger import Logger
from pytorchcheckpoint.checkpoint import CheckpointHandler
from rich.logging import RichHandler
from rich.progress import track

log = Logger(__name__).log
np.set_printoptions(precision=4, suppress=True, sign=" ")


class OkNetTrainer(Trainer):
    @torch.no_grad()
    def calculate_acc(self, dataloader, y_true: List = [], y_pred: List = []):
        """_summary_

        Parameters
        ----------
        dataloader : Dataloader
        y_true: List(Optional)
            Pass a list to this function to store the y_true values of the dataloader. This can be used to calculated
            the confusion matrix.
        y_pred: List(Optional)
            Pass a list to this function to store the y_pred of the dataloader.

        Returns
        -------
        acc: float
        pos_samples: int
            positive examples
        total: int
            total samples
        """
        acc_sum = 0
        pos_samples = 0
        total = 0
        for batch_idx, (x, y) in track(enumerate(dataloader), "Acc calculation: ", total=len(dataloader)):
            opt = x[0]
            kin = x[1]
            if self.gpu_boole:
                opt = opt.cuda()
                kin = kin.cuda()
                y = y.cuda()

            outputs = self.net((opt, kin))
            predictions = (outputs > 0.5).float()
            pos_samples += torch.sum(y).data.item()
            acc_sum += torch.sum(predictions == y)
            total += y.shape[0]

            y_pred += predictions.cpu().detach().numpy().squeeze().tolist()
            y_true += y.cpu().detach().numpy().squeeze().tolist()

        acc = acc_sum / total
        return acc.cpu().data.item(), pos_samples, total

    @torch.no_grad()
    def calculate_loss(self, dataloader: DataLoader):
        loss_sum = 0
        total = 0
        for batch_idx, (x, y) in track(enumerate(dataloader), "loss calculation: ", total=len(dataloader)):
            opt = x[0]
            kin = x[1]
            if self.gpu_boole:
                opt = opt.cuda()
                kin = kin.cuda()
                y = y.cuda()

            outputs = self.net((opt, kin))
            loss = self.loss_metric(outputs, y)
            loss_sum += loss * y.shape[0]
            total += y.shape[0]

        loss = loss_sum / total
        return loss.cpu().data.item()

    def train_loop(self, trial: optuna.Trial = None, verbose=True):
        log.info(f"Starting Training")
        valid_acc = 0
        for epoch in track(range(self.init_epoch, self.epochs), "Training network"):
            time1 = time.time()
            loss_sum = 0
            total = 0

            # Batch loop
            local_loss_sum = 0
            local_total = 0
            for batch_idx, (x, y) in enumerate(self.train_loader):
                opt = x[0]
                kin = x[1]
                if self.gpu_boole:
                    opt = opt.cuda()
                    kin = kin.cuda()
                    # x = x.cuda()
                    y = y.cuda()

                # loss calculation and gradient update:
                self.optimizer.zero_grad()
                outputs = self.net((opt, kin))
                loss = self.loss_metric(outputs, y)  # REMEMBER loss(OUTPUTS,LABELS)
                loss.backward()
                self.optimizer.step()  # Update parameters

                self.batch_count += 1
                # Global loss
                loss_sum += loss * y.shape[0]
                total += y.shape[0]
                # Local loss
                local_loss_sum += loss * y.shape[0]
                local_total += y.shape[0]

                if batch_idx % self.log_interval == 0:
                    local_loss = (local_loss_sum / local_total).cpu().item()
                    # End of batch stats
                    self.checkpoint_handler.store_running_var(
                        var_name="train_loss_batch", iteration=self.batch_count, value=local_loss
                    )
                    if verbose:
                        log.info(
                            f"epoch {epoch:3d} batch_idx {batch_idx:4d}/{len(self.train_loader)-1} local_loss {local_loss:0.6f}"
                        )

                    local_loss_sum = 0
                    local_total = 0

            # End of epoch statistics
            train_loss = loss_sum / total
            train_loss = train_loss.cpu().item()
            self.checkpoint_handler.store_running_var(var_name="train_loss", iteration=epoch, value=train_loss)

            # End of epoch additional metrics
            for m in self.end_of_epoch_metrics:
                self.epoch_metrics_dict[m].calc_new_value(**{"epoch": epoch})

            self.final_epoch = epoch
            self.init_epoch = self.final_epoch

            # Saving models
            if self.save:
                self.best_metric_to_opt = self.best_model_saver(self.checkpoint_handler, self.save_checkpoint)
                self.save_checkpoint("final_checkpoint.pt")

            # Print epoch information
            time2 = time.time()
            if verbose:
                log.info(f"*" * 30)
                log.info(f"Epoch {epoch}/{self.epochs-1}:")
                log.info(f"Elapsed time for epoch: { time2 - time1:0.04f} s")
                log.info(f"Training loss:     {train_loss:0.8f}")
                for m in self.end_of_epoch_metrics:
                    log.info(f"{m}: {self.epoch_metrics_dict[m].current_val:0.06f}")
                log.info(f"*" * 30)

            # Optune callbacks
            if self.optimize_hyperparams:
                # Optune prune mechanism
                trial.report(valid_acc, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return valid_acc
