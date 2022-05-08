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

from rich.progress import track
import time

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

    root = Config.trained_models_dir / "OKGAN/T3"
    if not root.exists():
        root.mkdir(parents=True)

    # opt_encoder_checkpt = Config.trained_models_dir / "encoder_decoder/T1/final_checkpoint.pt"

    # ------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------

    dataset = UnsupervisedBlobDatasetProbabilistic(blobs_folder_path=Config.blobs_dir)
    dataloader = DataLoader(dataset=dataset, batch_size=200, shuffle=False, collate_fn=size_collate_fn)

    # Model definition
    ok_net = OKNetV1(out_features=2048)# , reduce_kin_feat=True)
    ed_net = encoderDecoder(2048)


    correct_dataset = UnsupervisedBlobDatasetCorrect(blobs_folder_path=Config.blobs_dir)
    correct_dataloader = DataLoader(dataset=correct_dataset, batch_size=200, shuffle=True, collate_fn=size_collate_fn)
    # incorrect_dataset = UnsupervisedBlobDatasetIncorrect(blobs_folder_path=Config.blobs_dir)
    # dataset = ConcatDataset([correct_dataset, incorrect_dataset])
    # dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, collate_fn=size_collate_fn)
    # net = OKNetV2(out_features=2048)

    # ------------------------------------------------------------
    # Network and optimizer
    # ------------------------------------------------------------
    ok_loss_function = torch.nn.BCEWithLogitsLoss()
    ed_loss_function = torch.nn.MSELoss()

    oknet = ok_net.train()
    ednet = ed_net.train()
    if torch.cuda.is_available():
        oknet.cuda()
        ednet.cuda()

    ok_optimizer = torch.optim.Adam(params=oknet.parameters(), lr=lr, weight_decay=weight_decay)
    ed_optimizer = torch.optim.Adam(params=ednet.parameters(), lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------
    # Train Loop
    # ------------------------------------------------------------
    
    log.info(f"Starting Training")
    train_loss_values = []
    ed_loss_values = []
    ok_loss_values = []

    for epoch in track(range(num_epochs), "Training network"):
        time1 = time.time()
        loss_sum = 0
        ed_loss_sum = 0
        ok_loss_sum = 0
        total = 0

        # Batch loop
        local_loss_sum = 0
        local_total = 0
        for batch_idx, (x, y) in enumerate(correct_dataloader):
                
            opt = x[0]
            kin = x[1]
            if torch.cuda.is_available():
                opt = opt.cuda()
                kin = kin.cuda()
                # x = x.cuda()
                y = y.cuda()

            # loss calculation and gradient update:
            ed_optimizer.zero_grad()
            outputs1 = ed_net(opt)
            ed_loss = ed_loss_function(outputs1, kin)  # REMEMBER loss(OUTPUTS,LABELS)
            ed_loss.backward()
            ed_optimizer.step()  # Update parameters

            ok_optimizer.zero_grad()
            outputs2 = ok_net((opt, kin))
            ok_loss = ok_loss_function(outputs2, y)  # REMEMBER loss(OUTPUTS,LABELS)
            ok_loss.backward()
            ok_optimizer.step()  # Update parameters

            loss = ed_loss + ok_loss

            # Global loss
            loss_sum += loss * y.shape[0]
            ed_loss_sum += ed_loss * y.shape[0]
            ok_loss_sum += ok_loss * y.shape[0]
            total += y.shape[0]
            # Local loss
            local_loss_sum += loss * y.shape[0]
            local_total += y.shape[0]

            if batch_idx % 3 == 0:
                local_loss = (local_loss_sum / local_total).cpu().item()
                if True:
                    log.info(
                        f"epoch {epoch:3d} correct batch_idx {batch_idx:4d}/{len(dataloader)-1} local_loss {local_loss:0.6f}"
                    )

                local_loss_sum = 0
                local_total = 0
            
        for batch_idx, (x, y) in enumerate(dataloader):
            
            opt = x[0]
            kin = x[1]
            if torch.cuda.is_available():
                opt = opt.cuda()
                kin = kin.cuda()
                # x = x.cuda()
                y = y.cuda()

            # loss calculation and gradient update:
            # ed_optimizer.zero_grad()
            # outputs1 = ed_net(opt)
            # ed_loss = ed_loss_function(outputs1, kin)  # REMEMBER loss(OUTPUTS,LABELS)
            # ed_loss.backward()
            # ed_optimizer.step()  # Update parameters

            ok_optimizer.zero_grad()
            outputs2 = ok_net((opt, kin))
            ok_loss = ok_loss_function(outputs2, y)  # REMEMBER loss(OUTPUTS,LABELS)
            ok_loss.backward()
            ok_optimizer.step()  # Update parameters

            loss = ok_loss

            # Global loss
            loss_sum += loss * y.shape[0]
            ed_loss_sum += ed_loss * y.shape[0]
            ok_loss_sum += ok_loss * y.shape[0]
            total += y.shape[0]
            # Local loss
            local_loss_sum += loss * y.shape[0]
            local_total += y.shape[0]

            if batch_idx % 3 == 0:
                local_loss = (local_loss_sum / local_total).cpu().item()
                if True:
                    log.info(
                        f"epoch {epoch:3d} all batch_idx {batch_idx:4d}/{len(dataloader)-1} local_loss {local_loss:0.6f}"
                    )

                local_loss_sum = 0
                local_total = 0
            

        # End of epoch statistics
        train_loss = loss_sum / total
        train_loss = train_loss.cpu().item()

        # Append to the list for potting
        train_loss_values.append(train_loss)
        ed_loss_values.append((ed_loss / total). cpu().item())
        ok_loss_values.append((ok_loss / total). cpu().item())

        # Saving models
        if epoch == num_epochs-1:
            file_name_ok = root / "ok_final_checkpoint.pt"
            torch.save(oknet.state_dict(), file_name_ok)
            file_name_ed = root / "ed_final_checkpoint.pt"
            torch.save(ednet.state_dict(), file_name_ed)


        # Print epoch information
        time2 = time.time()
        if True:
            log.info(f"*" * 30)
            log.info(f"Epoch {epoch}/{num_epochs-1}:")
            log.info(f"Elapsed time for epoch: { time2 - time1:0.04f} s")
            log.info(f"Training loss:     {train_loss:0.8f}")
            log.info(f"*" * 30)

    # ------------------------------------------------------------
    # Result plots
    # ------------------------------------------------------------
    # Training plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    fig.suptitle("Epoch Training loss")
    X = np.arange(num_epochs)
    ax1.plot(X, train_loss_values, label="train_loss", color="blue")
    ax2.plot(X, ed_loss_values, label="ed_loss", color="blue")
    ax3.plot(X, ok_loss_values, label="ok_loss", color="blue")
    ax1.set_ylabel("Train Loss")
    ax2.set_ylabel("Encoder-Decoder Loss")
    ax3.set_ylabel("OKnet Loss")
    
    for ax in  (ax1, ax2, ax3):
        ax.set(xlabel="epoch")
        ax.legend()
        ax.grid()
    
    plt.savefig(root/'loss.png')
    plt.show()


def main():
    lr = 1e-3
    num_epochs = 200
    weights_save_path = "./weights_save"
    weight_decay = 1e-6

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