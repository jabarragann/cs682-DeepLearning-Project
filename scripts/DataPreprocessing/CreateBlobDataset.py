import numpy as np
import torch
import os
import pickle
from typing import Tuple, List
from torch.utils.data.dataloader import default_collate

from deepgesture.config import Config
from deepgesture.Dataset.BlobDataset import gestureBlobDataset, gestureBlobMultiDataset


def create_data_blobs(
    optical_flow_folder_path: str,
    transcriptions_folder_path: str,
    kinematics_folder_path: str,
    num_frames_per_blob: int,
    blobs_save_folder_path: str,
    spacing: int,
) -> None:
    if not os.path.exists(blobs_save_folder_path):
        os.makedirs(blobs_save_folder_path)

    blob_count = 0

    for file in os.listdir(transcriptions_folder_path):
        try:
            curr_file_path = os.path.join(transcriptions_folder_path, file)

            print("Processing file: {}".format(curr_file_path.split("/")[-1]))

            curr_optical_flow_file = "_".join([file.split(".")[0], "capture1.p"])
            curr_optical_flow_file = os.path.join(optical_flow_folder_path, curr_optical_flow_file)
            optical_flow_file = pickle.load(open(curr_optical_flow_file, "rb"))

            curr_kinematics_file = ".".join([file.split(".")[0], "txt"])
            curr_kinematics_file = os.path.join(kinematics_folder_path, curr_kinematics_file)
            kinematics_list = []

            with open(curr_kinematics_file) as kf:
                for line in kf:
                    kinematics_list.append([float(v) for v in line.strip("\n").strip().split("     ")])
                kf.close()

            with open(curr_file_path, "r") as f:  # Transcripts files
                for line in f:
                    line = line.strip("\n").strip()
                    line = line.split(" ")
                    start = int(line[0])
                    end = int(line[1])
                    gesture = line[2]

                    # optical_flow_file[0].shape --> (240,340,2) What is the dimension at the end?
                    curr_blob = [
                        torch.tensor(v)
                        for v in optical_flow_file[start : start + spacing * num_frames_per_blob : spacing]
                    ]
                    # This instruction seems odd. They are stacking the optical flow blob on top of each other.
                    curr_blob = torch.cat(curr_blob, dim=2).permute(2, 0, 1)
                    curr_kinematics_blob = [
                        torch.tensor(v).view(1, 76)
                        for v in kinematics_list[start : start + spacing * num_frames_per_blob : spacing]
                    ]
                    curr_kinematics_blob = torch.stack(curr_kinematics_blob, dim=0)
                    save_tuple = (curr_blob, curr_kinematics_blob)
                    curr_blob_save_path = (
                        "blob_"
                        + str(blob_count)
                        + "_video_"
                        + curr_file_path.split("/")[-1].split(".")[0].split("_")[-1]
                        + "_gesture_"
                        + gesture
                        + ".p"
                    )
                    curr_blob_save_path = os.path.join(blobs_save_folder_path, curr_blob_save_path)
                    pickle.dump(save_tuple, open(curr_blob_save_path, "wb"))

                    blob_count += 1
        except Exception as e:
            print(e)
            print("Erro in file: {}".format(curr_file_path.split("/")[-1]))


def size_collate_fn(batch: torch.Tensor) -> torch.Tensor:
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def main():
    optical_flow_folder_path = Config.optical_flow_dir
    transcriptions_folder_path = Config.transcriptions_dir
    num_frames_per_blob = 25
    blobs_save_folder_path = Config.blobs_dir
    spacing = 2
    kinematics_folder_path = Config.suturing_kinematics_dir

    create_data_blobs(
        optical_flow_folder_path=optical_flow_folder_path,
        transcriptions_folder_path=transcriptions_folder_path,
        kinematics_folder_path=kinematics_folder_path,
        num_frames_per_blob=num_frames_per_blob,
        blobs_save_folder_path=blobs_save_folder_path,
        spacing=spacing,
    )

    blobs_folder_paths_list = [blobs_save_folder_path]
    # dataset = gestureBlobDataset(blobs_folder_path = '../jigsaw_dataset/Knot_Tying/blobs/')
    dataset = gestureBlobMultiDataset(blobs_folder_paths_list=blobs_folder_paths_list)
    out = dataset.__getitem__(0)
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()
