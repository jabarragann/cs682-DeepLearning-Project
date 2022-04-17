import os
import cv2
import time
import numpy as np
from skimage.io import ImageCollection


scale = 0.25
epochs = 20
timestepsize = 10  # only get every n-th frame


class VideoLoader:
    def __init__(self):
        print("Creating video collection index...")
        self.viddir = "./dataset/Suturing/video/"
        labeldir = "./dataset/Suturing/transcriptions/"

        vidfiles = os.listdir(self.viddir)
        labelfiles = os.listdir(labeldir)
        maxframes = 0
        vidind = list()
        self.framelabels = list()
        for vid in vidfiles:
            video = cv2.VideoCapture(self.viddir + vid)
            minindex = maxframes
            maxframes += int(video.get(7))  # number of frames in this video
            vidind.append((maxframes, (vid, minindex)))
            video.release()

            # put together label sets for frames of this video
            with open(labeldir + vid[:-13] + ".txt") as f:
                t = f.read()
            lines = t.split(" \n")[:-1]
            vidlabelinfos = list(map(lambda x: x.split(' '), lines))
            vidlabels = [0] * (int(vidlabelinfos[0][0]) - 1)
            # 0 being the label of no gesture
            for labelinfo in vidlabelinfos:
                vidlabels += [int(labelinfo[2][1:])] * (int(labelinfo[1]) - int(labelinfo[0]) + 1)
            vidlabels += [0] * (maxframes - int(vidlabelinfos[-1][1]))
            self.framelabels += vidlabels
        self.framelabels = np.array(vidlabels)
        self.maxframe = maxframes - 1
        self.videoindex = vidind
        print("Index complete. Collection frame count: %s" % maxframes)

    def getVideoForFrame(self, frame):
        # returns a tuple with the video file name and the 0 frame number
        if frame > self.maxframe:
            print("[ERROR] Frame num too high for this collection")
            return None
        for index in self.videoindex:
            if frame < index[0]:
                return index[1]
        print("[ERROR] No video found for this frame???")
        return None

    def getFrame(self, frame,i):
        videoinfo = self.getVideoForFrame(frame)
        print(videoinfo)
        if videoinfo:
            framenum = frame - videoinfo[1]
            vid = cv2.VideoCapture(self.viddir + videoinfo[0])
            vid.set(1, framenum)
            ret, theframe = vid.read()
            vid.release()
            if ret == False:
                print("[ERROR] No frame returned")
                return None
            #modframe = cv2.resize(theframe, (int(480 * scale), int(640 * scale))) / 255.0
            cv2.imwrite("./dataset/SuturingImg/{}.jpg".format(i), theframe)

            #return np.swapaxes(np.swapaxes(modframe, 0, 2), 1, 2)
        print("[ERROR] No video info returned")
        return None


def main():


    # load dataset
    video_loader = VideoLoader()
    frame_pattern = range(0, video_loader.maxframe + 1, timestepsize)
    labels = video_loader.framelabels[frame_pattern]
    print(frame_pattern)
    frame = []
    for i in frame_pattern:
        frame.append(video_loader.getFrame(i,i))



    #ic = ImageCollection(frame)

    #print("Number of frames considered: %s" % len(ic))




if __name__ == "__main__":
    main()