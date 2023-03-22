''' A simple visualizing tool for making videos from frames '''
import os
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc
import cv2 as cv
from tqdm import tqdm

# Configuration of the video
width = 640
height = 480
FPS = 15

# Your video is stored at
fourcc = VideoWriter_fourcc(*'mp4v')
video = VideoWriter('/media/nistring/0f737a3f-358d-45e0-827c-473d6a7d555b/workspace/Abnormal-behavior-detection-in-ICU/AlphaPose/data/res2/yolox_fastpose_posex3d_short.mp4', fourcc, float(FPS), (width, height))
img_dir = '/media/nistring/0f737a3f-358d-45e0-827c-473d6a7d555b/workspace/Abnormal-behavior-detection-in-ICU/AlphaPose/data/res2/vis'

total_frames = len(os.listdir(img_dir))
for i in tqdm(range(total_frames)):
    frame = cv.imread(os.path.join(img_dir, str(i).zfill(5)+'.png'))
    video.write(frame)
video.release()