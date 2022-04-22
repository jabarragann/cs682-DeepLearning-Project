import numpy as np
import torch
from typing import Tuple
import torch.nn as nn

from torch.utils.data import DataLoader
from deepgesture.Dataset.BlobDataset import gestureBlobMultiDataset, size_collate_fn
from deepgesture.config import Config


class ConvNetStream(torch.nn.Module):
    def __init__(self, optical_flow_stream=False, out_features=512) -> None:
        super().__init__()
        if not optical_flow_stream:
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=2)
        else:
            self.conv1 = torch.nn.Conv2d(in_channels=2 * 25, out_channels=96, kernel_size=5, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1)
        self.conv4 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1)
        self.conv5 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1)

        self.linear1 = torch.nn.Linear(in_features=512 * 2 * 3, out_features=4096)
        self.linear2 = torch.nn.Linear(in_features=4096, out_features=out_features)
        self.dropout = torch.nn.Dropout(p=0.5)

        self.pool = torch.nn.MaxPool2d(2)

        self.softmax = torch.nn.Softmax(dim=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        # print('Shape of output after conv {} is {}'.format(1, x.size()))
        x = self.conv2(x)
        x = self.pool(x)
        # print('Shape of output after conv {} is {}'.format(2, x.size()))
        x = self.conv3(x)
        x = self.pool(x)
        # print('Shape of output after conv {} is {}'.format(3, x.size()))
        x = self.conv4(x)
        x = self.pool(x)
        # print(x.size())
        # print('Shape of output after conv {} is {}'.format(4, x.size()))
        # x = self.conv5(x)
        # x = self.pool(x)
        # print('Shape of output after conv {} is {}'.format(5, x.size()))

        x = x.view(-1, 512 * x.size()[2] * x.size()[3])
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.linear2(x)
        # # x = self.dropout(x)

        # x = self.softmax(x)
        return x


class twoStreamNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.spatial_net_stream = ConvNetStream(optical_flow_stream=False)
        self.temporal_net_stream = ConvNetStream(optical_flow_stream=True)

        self.linear1 = torch.nn.Linear(in_features=2 * 2048, out_features=512)
        # self.batch_norm = torch.nn.BatchNorm1d(512)
        self.linear2 = torch.nn.Linear(in_features=512, out_features=15)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: Tuple[torch.Tensor]) -> torch.Tensor:
        x1 = self.spatial_net_stream(x[0])
        x2 = self.temporal_net_stream(x[1])
        # print(x[0[.size())

        x_net = torch.cat((x1, x2), dim=1)
        x_net = self.linear1(x_net)

        x_net = self.linear2(x_net)

        x_net = self.softmax(x_net)
        return x_net


class encoderDecoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.conv_net_stream = ConvNetStream(optical_flow_stream=True, out_features=embedding_dim)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            torch.nn.Linear(128, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 25 * 76),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_net_stream(x)
        x = self.decoder(x)
        x = x.view(-1, 25, 1, 76)
        return x


if __name__ == "__main__":
    # load dataset
    gesture_dataset = gestureBlobMultiDataset(blobs_folder_paths_list=[Config.blobs_dir])
    dataloader = DataLoader(dataset=gesture_dataset, batch_size=128, shuffle=False, collate_fn=size_collate_fn)

    net = encoderDecoder(embedding_dim=2048)
    net = net.train()
    if torch.cuda.is_available():
        net.cuda()

    # Data accessing examples
    opt, kin = next(iter(dataloader))
    opt, kin = opt.cuda(), kin.cuda()
    out = net(opt)

    print(f"Output shape: {out.shape}")
    print(f"Optical flow: {opt.shape}")
    print(f"Kinematics:   {kin.shape}")
