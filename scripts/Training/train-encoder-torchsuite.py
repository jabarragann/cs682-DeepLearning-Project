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
import pytorchcheckpoint

# Project
from deepgesture.Dataset.BlobDataset import gestureBlobMultiDataset, size_collate_fn
from deepgesture.Models.EncoderDecoder import encoderDecoder
from deepgesture.config import Config


class TrainerNoAcc(Trainer):
    def calculate_acc(self, dataloader: DataLoader):
        return 0.0


def train_encoder_decoder_multidata_embeddings(
    lr: float, num_epochs: int, blobs_folder_paths_list: List[str], weights_save_path: str, weight_decay: float
) -> None:
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    if torch.cuda.is_available():
        print("Using CUDA")

    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)

    root = Config.trained_models_dir / "encoder_decoder/T1"
    if not root.exists():
        root.mkdir(parents=True)

    # ------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------
    gesture_dataset = gestureBlobMultiDataset(blobs_folder_paths_list=blobs_folder_paths_list)
    dataloader = DataLoader(dataset=gesture_dataset, batch_size=64, shuffle=False, collate_fn=size_collate_fn)

    # ------------------------------------------------------------
    # Network and optimizer
    # ------------------------------------------------------------
    loss_function = torch.nn.MSELoss()

    net = encoderDecoder(embedding_dim=2048)
    net = net.train()
    if torch.cuda.is_available():
        net.cuda()

    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------
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
    # checkpath = root / "best_checkpoint.pt"
    checkpath = root / "final_checkpoint.pt"
    if checkpath.exists():
        trainer_handler.load_checkpoint(checkpath)
        print(
            f"resuming training from epoch {trainer_handler.init_epoch}.\n Best metric: {trainer_handler.best_metric_to_opt:0.06f}"
        )

    loss_batch_store = trainer_handler.train_loop()

    # ------------------------------------------------------------
    # Result plots
    # ------------------------------------------------------------
    # Training plots
    training_board = TrainingBoard(trainer_handler.checkpoint_handler, root=root)
    training_board.training_plots()
    plt.show()


def main():
    lr = 1e-3
    num_epochs = 1000
    weights_save_path = "./weights_save"
    weight_decay = 1e-8

    blobs_folder_paths_list = [Config.blobs_dir]

    train_encoder_decoder_multidata_embeddings(
        lr=lr,
        num_epochs=num_epochs,
        blobs_folder_paths_list=blobs_folder_paths_list,
        weights_save_path=weights_save_path,
        weight_decay=weight_decay,
    )


if __name__ == "__main__":
    main()
