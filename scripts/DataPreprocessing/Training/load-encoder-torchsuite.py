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
from torch.utils.data import DataLoader, Dataset

# Custom trainer
import torchsuite
from torchsuite.Trainer import Trainer
from torchsuite.TrainingBoard import TrainingBoard
from pytorchcheckpoint.checkpoint import CheckpointHandler

# Project
from deepgesture.Dataset.BlobDataset import gestureBlobMultiDataset, size_collate_fn
from deepgesture.Models.EncoderDecoder import encoderDecoder
from deepgesture.config import Config


class TrainerNoAcc(Trainer):
    def calculate_acc(self, dataloader: DataLoader):
        return 0.0


def main():
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    lr = 1e-3
    num_epochs = 1000
    weight_decay = 1e-8

    if torch.cuda.is_available():
        print("Using CUDA")

    root = Config.trained_models_dir / "encoder_decoder/T1"
    if not root.exists():
        print(f"{root} does not contain a checkpoint")
        exit(0)

    # Load checkpoint
    net = encoderDecoder(embedding_dim=2048)
    net = net.cuda()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_function = torch.nn.MSELoss()
    checkpoint, net, optimizer = CheckpointHandler.load_checkpoint_with_model(
        root / "final_checkpoint.pt", net, optimizer
    )
    # Training plots
    training_board = TrainingBoard(checkpoint, root=root)
    training_board.training_plots()
    plt.show()

    # ------------------------------------------------------------
    # Test accuracy
    # ------------------------------------------------------------
    blobs_folder_paths_list = [Config.blobs_dir]
    gesture_dataset = gestureBlobMultiDataset(blobs_folder_paths_list=blobs_folder_paths_list)
    dataloader = DataLoader(dataset=gesture_dataset, batch_size=64, shuffle=False, collate_fn=size_collate_fn)
    trainer_handler = TrainerNoAcc(
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
    print("calculating loss...")
    train_loss = trainer_handler.calculate_loss(dataloader)
    print(f"Final training loss {train_loss:0.06f}")


if __name__ == "__main__":
    main()
