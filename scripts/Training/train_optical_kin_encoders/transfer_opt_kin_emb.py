"""
Train only the kinematic embedding. For this script you will need a pretrained optical flow encoder than can be loaded
to full model before training. 
"""

from pytorchcheckpoint.checkpoint import CheckpointHandler
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import classification_report
import os
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt

# Torch
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Custom trainer
import torchsuite
from torchsuite.Trainer import Trainer
from torchsuite.TrainingBoard import TrainingBoard
from torchsuite.utils.Logger import Logger
import pytorchcheckpoint

# Project
from deepgesture.Dataset.BlobDataset import gestureBlobMultiDataset, size_collate_fn
from deepgesture.Dataset.UnsuperviseBlobDataset import (
    UnsupervisedBlobDatasetProbabilistic,
    UnsupervisedBlobDatasetCorrect,
    UnsupervisedBlobDatasetIncorrect,
)
from deepgesture.Models.EncoderDecoder import encoderDecoder
from deepgesture.Models.OpticalFlowKinematicEncoder import OKNetV1, OKNetV2
from deepgesture.config import Config

from OkNetworkTrainer import OkNetTrainer
from deepgesture.Models.EncoderDecoder import encoderDecoder

log = Logger("train_emb").log


def calculate_encoder_decoder_loss(net: nn.Module, gpu_boole: bool = True):
    # Dataset
    blobs_folder_paths_list = [Config.blobs_dir]
    gesture_dataset = gestureBlobMultiDataset(blobs_folder_paths_list=blobs_folder_paths_list)
    dataloader = DataLoader(dataset=gesture_dataset, batch_size=64, shuffle=False, collate_fn=size_collate_fn)

    loss_metric = torch.nn.MSELoss()
    loss_sum = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            if gpu_boole:
                x = x.cuda()
                y = y.cuda()
            outputs = net(x)
            loss_sum += loss_metric(outputs, y) * y.shape[0]
            total += y.shape[0]
        loss = loss_sum / total
    return loss.cpu().data.item()


def train_kin_embedding(
    lr: float, num_epochs: int, blobs_folder_paths_list: List[str], weights_save_path: str, weight_decay: float
) -> None:
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    if torch.cuda.is_available():
        print("Using CUDA")

    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)

    root = Config.trained_models_dir / "ok_network/T10"
    if not root.exists():
        root.mkdir(parents=True)

    opt_encoder_checkpt = Config.trained_models_dir / "encoder_decoder/T1/final_checkpoint.pt"
    # ------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------

    dataset = UnsupervisedBlobDatasetProbabilistic(blobs_folder_path=Config.blobs_dir)
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False, collate_fn=size_collate_fn)
    net = OKNetV1(out_features=2048)# , reduce_kin_feat=True)

    # correct_dataset = UnsupervisedBlobDatasetCorrect(blobs_folder_path=Config.blobs_dir)
    # incorrect_dataset = UnsupervisedBlobDatasetIncorrect(blobs_folder_path=Config.blobs_dir)
    # dataset = ConcatDataset([correct_dataset, incorrect_dataset])
    # dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, collate_fn=size_collate_fn)
    # net = OKNetV2(out_features=2048)

    # ------------------------------------------------------------
    # Network and optimizer
    # ------------------------------------------------------------
    loss_function = torch.nn.BCEWithLogitsLoss()
    # loss_function = torch.nn.BCELoss()

    net = net.train()
    if torch.cuda.is_available():
        net.cuda()

    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------
    # Load model optical flow embedding - Transfer learning step
    # ------------------------------------------------------------
    checkpoint_handler: CheckpointHandler = CheckpointHandler.load_checkpoint(opt_encoder_checkpt)
    encoder_decoder_state = checkpoint_handler.model_state_dict
    encoder_decoder_net = encoderDecoder(2048)
    encoder_decoder_net.load_state_dict(encoder_decoder_state)
    encoder_decoder_net = encoder_decoder_net.cuda()

    # log.info("calculating loss...")
    # train_loss = calculate_encoder_decoder_loss(encoder_decoder_net)
    # log.info(f"train loss {train_loss:0.4f}")

    opt_encoder = encoder_decoder_net.conv_net_stream
    for param in opt_encoder.parameters():
        param.requires_grad = False # false

    net.opticalflow_net_stream = opt_encoder

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
    net.app

    total_params = sum(p.numel() for p in net.parameters())
    log.info(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log.info(f"{total_trainable_params:,} training parameters in net.")
    opt_enc_trainable_params = sum(p.numel() for p in opt_encoder.parameters() if p.requires_grad)
    log.info(f"{opt_enc_trainable_params:,} training parameters in opt_encoder.")
    log.info(f"%of trainable params in net {total_trainable_params/total_params*100:0.4f}%")

    # ------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------
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
        log_interval=3,
        end_of_epoch_metrics=[],  # ["train_acc", "valid_acc"]
    )
    # checkpath = root / "best_checkpoint.pt"
    # checkpath = root / "final_checkpoint.pt"
    # if checkpath.exists():
    #     trainer_handler.load_checkpoint(checkpath)
    #     print(
    #         f"resuming training from epoch {trainer_handler.init_epoch}.\n Best metric: {trainer_handler.best_metric_to_opt:0.06f}"
    #     )

    loss_batch_store = trainer_handler.train_loop()

    # ------------------------------------------------------------
    # Result plots
    # ------------------------------------------------------------
    # Training plots
    training_board = TrainingBoard(trainer_handler.checkpoint_handler, root=root)
    training_board.training_plots()
    plt.show()


def main():
    lr = 1e-7
    num_epochs = 1500
    weights_save_path = "./weights_save"
    weight_decay = 1e-8

    blobs_folder_paths_list = [Config.blobs_dir]

    train_kin_embedding(
        lr=lr,
        num_epochs=num_epochs,
        blobs_folder_paths_list=blobs_folder_paths_list,
        weights_save_path=weights_save_path,
        weight_decay=weight_decay,
    )


if __name__ == "__main__":
    main()