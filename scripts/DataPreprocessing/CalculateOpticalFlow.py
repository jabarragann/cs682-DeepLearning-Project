import numpy as np
import cv2
import os
import pickle
import sys
from tqdm import tqdm
from typing import Tuple
import torch

from deepgesture.config import Config


class computeOpticalFlow:
    def __init__(
        self,
        source_directory: str,
        resized_video_directory: str = "None",
        destination_directory: str = "./optical_flows",
        resize_dim: Tuple[int, int] = (320, 240),
    ) -> None:
        self.source_directory = source_directory
        self.destination_directory = destination_directory
        self.resized_video_directory = resized_video_directory

        if not os.path.exists(self.destination_directory):
            os.makedirs(self.destination_directory)

        self.source_listdir = os.listdir(self.source_directory)
        self.source_listdir = list(filter(lambda x: ".DS_Store" not in x, self.source_listdir))
        self.source_listdir.sort()
        self.resize_dim = resize_dim

        self.resized_listdir = None

    def get_optical_frame(self):
        for video in self.resized_listdir:
            print("Processing video {}".format(video))
            curr_video_path = os.path.join(self.resized_video_directory, video)
            vidcap = cv2.VideoCapture(curr_video_path)

            success, image = vidcap.read()

            prvs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            m, n = prvs.shape
            # prvs = cv2.resize(prvs, (int(n), int(m)))
            prvs = cv2.resize(prvs, self.resize_dim)
            temp_optical_flow_frames = []

            count = 1

            while success:
                success, image = vidcap.read()
                try:
                    _next = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    _next = cv2.resize(_next, self.resize_dim)
                    flow = cv2.calcOpticalFlowFarneback(prvs, _next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    temp_optical_flow_frames.append(flow)
                    prvs = _next
                    count += 1
                except:
                    pass
                if count % 100 == 0:
                    print(count)

            print("Saving file")
            save_path = os.path.join(self.destination_directory, video.split(".")[0] + ".p")
            pickle.dump(temp_optical_flow_frames, open(save_path, "wb"))

    def rescale_video(self) -> None:
        if not os.path.exists(self.resized_video_directory):
            os.makedirs(self.resized_video_directory)

        for video in self.source_listdir:
            print("Processing video {}".format(video))
            curr_video_path = os.path.join(self.source_directory, video)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            vidcap = cv2.VideoCapture(curr_video_path)
            output_path = os.path.join(self.resized_video_directory, video.split(".")[0] + "_resized.avi")
            out = cv2.VideoWriter(output_path, fourcc, 20.0, self.resize_dim)

            success, image = vidcap.read()

            count = 1

            while success:
                success, image = vidcap.read()
                try:
                    resized_image = cv2.resize(image, self.resize_dim)
                    out.write(resized_image)
                    count += 1
                except:
                    pass

                if count % 100 == 0:
                    print(count)

            vidcap.release()
            out.release()

    def run(self) -> None:
        if self.resized_video_directory == "None":
            self.resized_listdir = self.source_listdir
            self.resized_video_directory = self.source_directory
            print("Generating optical flow.")
            self.get_optical_frame()
        else:
            print("Rescaling videos.")
            self.rescale_video()
            self.resized_listdir = os.listdir(self.resized_video_directory)
            self.resized_listdir = list(filter(lambda x: ".DS_Store" not in x, self.resized_listdir))
            self.resized_listdir.sort()
            print("Generating optical flow.")
            self.get_optical_frame()


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

            curr_optical_flow_file = "_".join([file.split(".")[0], "capture1_resized.p"])
            curr_optical_flow_file = os.path.join(optical_flow_folder_path, curr_optical_flow_file)
            optical_flow_file = pickle.load(open(curr_optical_flow_file, "rb"))

            curr_kinematics_file = ".".join([file.split(".")[0], "txt"])
            curr_kinematics_file = os.path.join(kinematics_folder_path, curr_kinematics_file)
            kinematics_list = []

            with open(curr_kinematics_file) as kf:
                for line in kf:
                    kinematics_list.append([float(v) for v in line.strip("\n").strip().split("     ")])
                kf.close()

            with open(curr_file_path, "r") as f:
                for line in f:
                    line = line.strip("\n").strip()
                    line = line.split(" ")
                    start = int(line[0])
                    end = int(line[1])
                    gesture = line[2]
                    curr_blob = [
                        torch.tensor(v)
                        for v in optical_flow_file[start : start + spacing * num_frames_per_blob : spacing]
                    ]
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
        except:
            pass


def main():
    source_directory = Config.suturing_videos_dir  
    # resized_video_directory = '../jigsaw_dataset/Surgeon_study_videos/resized_videos'
    resized_video_directory = "None"
    destination_directory = Config.optical_flow_dir 
    resize_dim = (320, 240)

    optical_flow_compute = computeOpticalFlow(
        source_directory=source_directory,
        resized_video_directory=resized_video_directory,
        destination_directory=destination_directory,
        resize_dim=resize_dim,
    )
    optical_flow_compute.run()
    print(optical_flow_compute.source_listdir)


if __name__ == "__main__":
    main()
