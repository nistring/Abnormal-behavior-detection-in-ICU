

import glob
import os
import time
from pathlib import Path
from threading import Thread
import multiprocessing as mp
import zmq
import cv2
import numpy as np
import h5py
from copy import deepcopy
import pickle

from EtherSense.EtherSenseClient import main, D2RGB, RGB2D

class LoadWebcam:  # for inference
    def __init__(self, img_size=640, stride=32):
        mp.Process(target=main, args=()).start()
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect('tcp://127.0.0.1:5555')

        self.img_size = img_size
        self.stride = stride
        self.cap = None

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        self.socket.send(b"msg")
        data = pickle.loads(self.socket.recv())
        path = data.keys()
        img0 = np.stack(data.values())
        img = np.stack([letterbox(img0[i], self.img_size, stride=self.stride)[0] for i in range(self.num_servers)])
        

        #path = str(self.count)
        box = None
        
        self.count += 1

        return path, img, img0, depth, box, keypoints, self.cap
        
    def __len__(self):
        return 0
    
def letterbox(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular 
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height)/shape[0], float(width)/shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio)) # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh
