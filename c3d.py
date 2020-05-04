import cv2
import numpy as np


class Args:
    batch = 64
    epoch = 100
    nclass = 7  # 11 action categories
    depth = 10
    rows = 32
    cols = 32
    skip = True  # Skip: randomly extract frames; otherwise, extract first few frames
    channel = 1


class Videoto3D:

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def get_data(self, filename, skip=True):
        cap = cv2.VideoCapture(filename)
        nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        bAppend = False
        if (nframe >= self.depth):
            if skip:
                frames = [x * nframe / self.depth for x in range(self.depth)]
            else:
                frames = [x for x in range(self.depth)]
        else:
            print("Insufficient %d frames in video %s, set bAppend as True" % (nframe, filename))
            bAppend = True
            frames = [x for x in range(int(nframe))]  # nframe is a float

        framearray = []

        for i in range(len(frames)):  # self.depth):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
            ret, frame = cap.read()
            frame = cv2.resize(frame, (self.height, self.width))
            framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        cap.release()

        if bAppend:
            while len(framearray) < self.depth:
                framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            print("Append more frames in the framearray to have %d frames" % len(framearray))

        return np.array(framearray)


def loaddata(video_list, vid3d, skip=True):
    X = []
    Y = []
    for idx, value in enumerate(video_list):
        # Display the progress
        if (idx % 100) == 0:
            print("process data %d/%d" % (idx, len(video_list)))
        filename = value[0]
        label = value[1]
        Y.append(label)
        X.append(vid3d.get_data(filename, skip=skip))

    return np.array(X).transpose((0, 2, 3, 1)), np.array(Y)


def load_testdata(file, vid3d):
    res = vid3d.load_data(file, skip=True)
    res = [res]
    res = np.array(res).transpose((0, 2, 3, 1))
    res = res.reshape((res.shape[0], Args.rows, Args.rows.cols, Args.depth, Args.channel))

    return res
