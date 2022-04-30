import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import os
from datetime import datetime
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt

# Torch
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset

# Custom trainer
import torchsuite
from torchsuite.Trainer import Trainer
from torchsuite.TrainingBoard import TrainingBoard
from pytorchcheckpoint.checkpoint import CheckpointHandler

# Project
from deepgesture.Dataset.BlobDataset import gestureBlobMultiDataset, size_collate_fn
from deepgesture.Dataset.UnsuperviseBlobDataset import (
    UnsupervisedBlobDatasetCorrect,
    UnsupervisedBlobDatasetIncorrect,
    UnsupervisedBlobDatasetProbabilistic,
)
from deepgesture.Models.EncoderDecoder import encoderDecoder
from deepgesture.Models.OpticalFlowKinematicEncoder import OKNetV1, OKNetV2
from deepgesture.config import Config

from OkNetworkTrainer import OkNetTrainer
from torchsuite.utils.Logger import Logger

log = Logger("ok network load").log


def main():
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    lr = 1e-3
    num_epochs = 1000
    weight_decay = 1e-8

    if torch.cuda.is_available():
        print("Using CUDA")

    root = Config.trained_models_dir / "ok_network/T2"
    if not root.exists():
        print(f"{root} does not contain a checkpoint")
        exit(0)

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    ## Probabilistic dataset
    # dataset = UnsupervisedBlobDatasetProbabilistic(blobs_folder_path=Config.blobs_dir)
    # dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=False, collate_fn=size_collate_fn)
    # net = OKNetV1(out_features=2048)

    # Complete dataset
    correct_dataset = UnsupervisedBlobDatasetCorrect(blobs_folder_path=Config.blobs_dir)
    incorrect_dataset = UnsupervisedBlobDatasetIncorrect(blobs_folder_path=Config.blobs_dir)
    dataset = ConcatDataset([correct_dataset, incorrect_dataset])
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, collate_fn=size_collate_fn)

    log.info(f"Correct dataset   {len(correct_dataset)}")
    log.info(f"Incorrect dataset {len(correct_dataset)}")
    net = OKNetV2(out_features=2048)

    # Load checkpoint
    # net = net.eval()
    net = net.train()
    net = net.cuda()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = torch.nn.BCEWithLogitsLoss()
    checkpoint, net, optimizer = CheckpointHandler.load_checkpoint_with_model(
        root / "final_checkpoint.pt", net, optimizer
    )
    # Training plots
    training_board = TrainingBoard(checkpoint, root=root)
    training_board.training_plots()
    plt.show()

    log.info(f"total dataset     {len(dataset)}")

    trainer_handler = OkNetTrainer(
        dataloader,
        None,
        net,
        optimizer,
        loss_function,
        num_epochs,
        root=root,
        gpu_boole=True,
        save=True,
        log_interval=2,
        end_of_epoch_metrics=[],  # ["train_acc", "valid_acc"]
    )

    train_loss = trainer_handler.calculate_loss(dataloader)
    log.info(f"Final training loss {train_loss:0.06f}")
    train_acc, pos_samples, total_samples = trainer_handler.calculate_acc(dataloader)
    log.info(f"Final training acc {train_acc:0.06f}")

    log.info("DATASET STATS")
    log.info(f"Total samples  {total_samples}")
    log.info(f"Total positive {pos_samples}")
    log.info(f"Total negative {total_samples-pos_samples}")

    # import pdb
    # pdb.set_trace()


if __name__ == "__main__":
    main()
