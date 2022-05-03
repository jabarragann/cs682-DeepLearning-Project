from pathlib import Path
import sys
import numpy as np
import torch
import os
import pickle
from typing import Tuple, List
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, ConcatDataset
from deepgesture.config import Config
import random
import math as m
from abc import ABC, abstractmethod


class UnsupervisedBlobDatasetAbstract(ABC):
    """Unsupervised dataset abstract template"""

    def __init__(self, blobs_folder_path: str) -> None:
        self.blobs_folder_path = blobs_folder_path
        self.blobs_folder = os.listdir(self.blobs_folder_path)
        self.blobs_folder = list(filter(lambda x: ".DS_Store" not in x, self.blobs_folder))
        self.blobs_folder.sort(key=lambda x: int(x.split("_")[1]))

    def __len__(self) -> int:
        return len(self.blobs_folder)

    @abstractmethod
    def __getitem__(self, idx: int) -> torch.Tensor:
        pass


class UnsupervisedBlobDatasetProbabilistic(UnsupervisedBlobDatasetAbstract):
    """Dataset for unsupervised training of optical and kinematic encoders.
    When accessing a sample the kinematic data will be replace with the data from another
    sample with a probability of 0.5. The label will be either 1 if the kinematic and optical
    flow correspond to the same sample or 0 otherwise.
    """

    def __init__(self, blobs_folder_path: str) -> None:
        super().__init__(blobs_folder_path)

    def __getitem__(self, idx: int) -> torch.Tensor:
        curr_file_path = self.blobs_folder[idx]
        curr_file_path = os.path.join(self.blobs_folder_path, curr_file_path)
        curr_tensor_tuple = pickle.load(open(curr_file_path, "rb"))
        # print(curr_tensor_tuple[0].size())

        if curr_tensor_tuple[0].size()[0] != 50:
            print(f"error in {curr_file_path}")
            return None

        # Get random number between 0 and 1
        p = random.random()
        if p > 0.5:
            return curr_tensor_tuple, torch.tensor([1.0], dtype=torch.float32)
        else:
            # Load random blob
            new_idx = int(m.floor(len(self) * p))
            if new_idx == idx:
                new_idx += 1
            if new_idx == len(self):
                return None
            curr_file_path = os.path.join(self.blobs_folder_path, self.blobs_folder[new_idx])
            new_tensor_tuple = pickle.load(open(curr_file_path, "rb"))
            # Swap kinematic data between blobs
            combined_tuple = (curr_tensor_tuple[0], new_tensor_tuple[1])

            return combined_tuple, torch.tensor([0.0], dtype=torch.float32)


class UnsupervisedBlobDatasetCorrect(UnsupervisedBlobDatasetAbstract):
    """Dataset for unsupervised training of optical and kinematic encoders.
    Samples from this dataset will have the correct kinematic and optical flow data.
    """

    def __init__(self, blobs_folder_path: str) -> None:
        super().__init__(blobs_folder_path)

    def __getitem__(self, idx: int) -> torch.Tensor:
        curr_file_path = self.blobs_folder[idx]
        curr_file_path = os.path.join(self.blobs_folder_path, curr_file_path)
        curr_tensor_tuple = pickle.load(open(curr_file_path, "rb"))
        # print(curr_tensor_tuple[0].size())

        if curr_tensor_tuple[0].size()[0] != 50:
            print(f"error in {curr_file_path}")
            return None

        return curr_tensor_tuple, torch.tensor([1.0], dtype=torch.float32)


class UnsupervisedBlobDatasetIncorrect(UnsupervisedBlobDatasetAbstract):
    """Dataset for unsupervised training of optical and kinematic encoders.
    Samples from this dataset will have the incorrect kinematic and optical flow data.
    """

    def __init__(self, blobs_folder_path: str) -> None:
        super().__init__(blobs_folder_path)

    def __getitem__(self, idx: int) -> torch.Tensor:
        curr_file_path = self.blobs_folder[idx]
        curr_file_path = os.path.join(self.blobs_folder_path, curr_file_path)
        curr_tensor_tuple = pickle.load(open(curr_file_path, "rb"))
        # print(curr_tensor_tuple[0].size())

        if curr_tensor_tuple[0].size()[0] != 50:
            print(f"error in {curr_file_path}")
            return None

        # Get random number between 0 and 1
        p = random.random()
        new_idx = int(m.floor(len(self) * p))
        if new_idx == idx:
            new_idx += 1
        if new_idx == len(self):
            return None

        curr_file_path = os.path.join(self.blobs_folder_path, self.blobs_folder[new_idx])
        new_tensor_tuple = pickle.load(open(curr_file_path, "rb"))
        # Swap kinematic data between blobs
        combined_tuple = (curr_tensor_tuple[0], new_tensor_tuple[1])

        return combined_tuple, torch.tensor([0.0], dtype=torch.float32)


def size_collate_fn(batch: torch.Tensor) -> torch.Tensor:
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def main():
    num_frames_per_blob = 25
    spacing = 2
    blobs_save_folder_path = Config.blobs_dir

    dataset = UnsupervisedBlobDatasetProbabilistic(blobs_folder_path=Config.blobs_dir)
    dataloader = DataLoader(dataset=dataset, batch_size=24, shuffle=False, collate_fn=size_collate_fn)

    # Data accessing examples
    print(f"Length of dataset: {len(dataset)}")
    out, label = dataset.__getitem__(8)
    print(f"Label {label}")
    print(f"Optical flow shape: {out[0].shape}")
    print(f"Kinematics data shape: {out[1].shape}")

    (opt, kin), label = next(iter(dataloader))
    print(f"labels {label}")
    print(f"Optical flow: {opt.shape}")
    print(f"Kinematics: {kin.shape}")

    # Correct incorrect dataset
    correct_dataset = UnsupervisedBlobDatasetCorrect(blobs_folder_path=Config.blobs_dir)
    incorrect_dataset = UnsupervisedBlobDatasetIncorrect(blobs_folder_path=Config.blobs_dir)
    dataset = ConcatDataset([correct_dataset, incorrect_dataset])
    dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, collate_fn=size_collate_fn)

    (opt, kin), label = next(iter(dataloader))
    print(f"labels {label}")
    print(f"Optical flow: {opt.shape}")
    print(f"Kinematics: {kin.shape}")


if __name__ == "__main__":
    main()
