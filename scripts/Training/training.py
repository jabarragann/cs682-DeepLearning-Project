from distutils.command.config import config
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import os
from datetime import datetime
from typing import Tuple, List

from deepgesture.Dataset.BlobDataset import gestureBlobMultiDataset, size_collate_fn
from deepgesture.Models.EncoderDecoder import encoderDecoder
from deepgesture.config import Config

# Use subset sampler for train test split


def calc_labels(y: torch.Tensor) -> torch.Tensor:
    # out = torch.ones(y.size()[0], 15, dtype = torch.float32)*(0.01/14)
    out = torch.zeros(y.size()[0], 15, dtype=torch.long)
    # print(out.size())
    for i in range(out.size()[0]):
        # out[i, int(y[i].item()) - 1] = 0.99
        out[i, int(y[i].item()) - 1] = 1
    return out


def train_multimodal_embeddings(
    lr: float, num_epochs: int, blobs_folder_path: str, weights_save_path: str, weight_decay: float
) -> None:
    if torch.cuda.is_available():
        print("Using CUDA")

    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)

    gesture_dataset = gestureBlobDataset(blobs_folder_path=blobs_folder_path)
    dataset = gestureBlobBatchDataset(gesture_dataset=gesture_dataset, random_tensor="random")
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

    loss_function = torch.nn.MSELoss()
    # loss_function = torch.nn.KLDivLoss()
    # net = multiModalRepresentation_diff(out_features = 2048, lstm_num_layers= 2, parser = 'cnn')
    net = multiModalRepresentation(out_features=512, lstm_num_layers=2, parser="cnn")
    net = net.train()
    if torch.cuda.is_available():
        net.cuda()

    # Remove comment to initialized at a pretrained point
    # net.load_state_dict(torch.load(os.path.join(weights_save_path,'two_stream_Knot_Tying_2020-02-01_20:16:16.pth' )))

    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        running_loss = 0
        count = 0
        print("Epoch {}".format(epoch + 1))
        # for idx in range(len(dataset)):
        for data in dataloader:
            current_tensor, random_tensor, y_match, y_rand = data
            curr_opt, curr_kin = current_tensor
            _, rand_kin = random_tensor
            # kin = torch.cat([kinematics, kinematics_rand], dim = 1)
            if torch.cuda.is_available():
                curr_opt = curr_opt.cuda()
                curr_kin = curr_kin.cuda()
                y_match = y_match.cuda()
                rand_kin = rand_kin.cuda()
                y_rand = y_rand.cuda()
            optimizer.zero_grad()
            # import pdb; pdb.set_trace()
            out1 = net((curr_opt, curr_kin))
            out2 = net((curr_opt, rand_kin))
            loss = loss_function(out2, y_rand) + loss_function(out1, y_match)
            # loss = loss_function(out2.log(), y_rand) + loss_function(out1.log(), y_match) # Use for KLDivLoss
            loss.backward()
            optimizer.step()
            print("Out1: {}".format(out1[0, :]))
            print("Out2: {}".format(out2[0, :]))
            print("Current loss2 = {}".format(loss.item()))
            running_loss += loss.item()
            count += 1
        print(
            out1[
                0,
                :,
            ]
        )
        print(out2[0, :])
        print("\n Epoch: {}, Loss: {}".format(epoch + 1, running_loss / count))

    print("Finished training.")
    print("Saving state dict.")

    now = datetime.now()
    now = "_".join((str(now).split(".")[0]).split(" "))
    file_name = "multimodal_" + dataset_name + "_" + now + ".pth"
    file_name = os.path.join(weights_save_path, file_name)
    torch.save(net.state_dict(), file_name)

    print("State dict saved at timestamp {}".format(now))


def train_encoder_decoder_embeddings(
    lr: float, num_epochs: int, blobs_folder_path: str, weights_save_path: str, weight_decay: float, dataset_name: str
) -> None:
    if torch.cuda.is_available():
        print("Using CUDA")

    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)

    gesture_dataset = gestureBlobDataset(blobs_folder_path=blobs_folder_path)
    dataloader = DataLoader(dataset=gesture_dataset, batch_size=128, shuffle=False, collate_fn=size_collate_fn)

    loss_function = torch.nn.MSELoss()
    # loss_function = torch.nn.KLDivLoss()
    net = encoderDecoder(embedding_dim=512)
    net = net.train()
    if torch.cuda.is_available():
        net.cuda()

    # Remove comment to initialized at a pretrained point
    # net.load_state_dict(torch.load(os.path.join(weights_save_path,'two_stream_Knot_Tying_2020-02-01_20:16:16.pth' )))

    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        running_loss = 0
        count = 0
        print("Epoch {}".format(epoch + 1))
        for data in dataloader:
            curr_opt, curr_kin = data
            if torch.cuda.is_available():
                curr_opt = curr_opt.cuda()
                curr_kin = curr_kin.cuda()
            optimizer.zero_grad()
            out1 = net(curr_opt)
            loss = loss_function(out1, curr_kin)
            loss.backward()
            optimizer.step()
            # print('Current loss = {}'.format(loss.item()))
            running_loss += loss.item()
            count += 1
        print("\n Epoch: {}, Loss: {}".format(epoch + 1, running_loss / count))

    print("Finished training.")
    print("Saving state dict.")

    now = datetime.now()
    now = "_".join((str(now).split(".")[0]).split(" "))
    file_name = "multimodal_" + dataset_name + "_" + now + ".pth"
    file_name = os.path.join(weights_save_path, file_name)
    torch.save(net.state_dict(), file_name)

    print("State dict saved at timestamp {}".format(now))


def train_encoder_decoder_multidata_embeddings(
    lr: float, num_epochs: int, blobs_folder_paths_list: List[str], weights_save_path: str, weight_decay: float
) -> None:
    if torch.cuda.is_available():
        print("Using CUDA")

    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)

    gesture_dataset = gestureBlobMultiDataset(blobs_folder_paths_list=blobs_folder_paths_list)
    dataloader = DataLoader(dataset=gesture_dataset, batch_size=64, shuffle=False, collate_fn=size_collate_fn)

    loss_function = torch.nn.MSELoss()
    # loss_function = torch.nn.KLDivLoss()
    net = encoderDecoder(embedding_dim=2048)
    net = net.train()
    if torch.cuda.is_available():
        net.cuda()

    # Remove comment to initialized at a pretrained point
    # net.load_state_dict(torch.load(os.path.join(weights_save_path,'two_stream_Knot_Tying_2020-02-01_20:16:16.pth' )))

    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        running_loss = 0
        count = 0
        print("Epoch {}".format(epoch + 1))
        for data in dataloader:
            curr_opt, curr_kin = data
            if torch.cuda.is_available():
                curr_opt = curr_opt.cuda()
                curr_kin = curr_kin.cuda()
            optimizer.zero_grad()
            out1 = net(curr_opt)
            loss = loss_function(out1, curr_kin)
            loss.backward()
            optimizer.step()
            # print('Current loss = {}'.format(loss.item()))
            running_loss += loss.item()
            count += 1
        print("\n Epoch: {}, Loss: {}".format(epoch + 1, running_loss / count))

    print("Finished training.")
    print("Saving state dict.")

    now = datetime.now()
    now = "_".join((str(now).split(".")[0]).split(" "))
    file_name = "multimodal_multidata_" + now + ".pth"
    file_name = os.path.join(weights_save_path, file_name)
    torch.save(net.state_dict(), file_name)

    print("State dict saved at timestamp {}".format(now))


def main():
    blobs_folder_path = "../jigsaw_dataset/Knot_Tying/blobs"
    lr = 1e-3
    num_epochs = 1000
    weights_save_path = "./weights_save"
    weight_decay = 1e-8
    dataset_name = "Knot_Tying"

    blobs_folder_paths_list = [Config.blobs_dir]

    # train_multimodal_embeddings(lr = lr, num_epochs = num_epochs, blobs_folder_path = blobs_folder_path, weights_save_path = weights_save_path, weight_decay = weight_decay)
    # train_encoder_decoder_embeddings(lr = lr, num_epochs = num_epochs, blobs_folder_path = blobs_folder_path, weights_save_path = weights_save_path, weight_decay = weight_decay, dataset_name = dataset_name)
    train_encoder_decoder_multidata_embeddings(
        lr=lr,
        num_epochs=num_epochs,
        blobs_folder_paths_list=blobs_folder_paths_list,
        weights_save_path=weights_save_path,
        weight_decay=weight_decay,
    )


if __name__ == "__main__":
    main()
