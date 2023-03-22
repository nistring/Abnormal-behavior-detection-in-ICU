import glob
import os
import time
from pathlib import Path
from threading import Thread
import multiprocessing as mp

import cv2
import numpy as np
import h5py
from copy import deepcopy

from EtherSense.EtherSenseClient import multi_cast_message, D2RGB, RGB2D

class LoadWebcam:  # for inference
    def __init__(self, num_servers : int, img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.cap = None
        self.num_servers = num_servers

        self.queues = [mp.Queue(maxsize=3) for _ in range(num_servers)]
        self.lock = mp.Lock()
        process = mp.Process(target=multi_cast_message, args=("EtherSense ping", num_servers, self.lock, self.queues))
        process.start()
        time.sleep(5)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):

        path = str(self.count)
        box = None
        keypoints = None

        while True:
            with self.lock:
                if all([not queue.empty() for queue in self.queues]):
                    depth = np.stack([queue.get() for queue in self.queues], axis=0)
                    break
            time.sleep(0.01)

        img0 = D2RGB(depth)

        # Padded resize
        img = np.stack([letterbox(img0[i], self.img_size, stride=self.stride)[0] for i in range(self.num_servers)])

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2) # BGR to RGB, to 1x3x416x416
        img = np.ascontiguousarray(img)

        self.count += 1

        return path, img, img0, depth, box, keypoints, self.cap
        
    def __len__(self):
        return 0