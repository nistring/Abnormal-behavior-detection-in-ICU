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

from yolov7.utils.general import check_requirements, clean_str
from yolov7.utils.datasets import letterbox

from EtherSense.EtherSenseClient import multi_cast_message, D2RGB, RGB2D


# Parameters
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
depth_formats = ['npy']

class LoadImages:  # for inference
    def __init__(self, path, annotations, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        depths = [x for x in files if x.split('.')[-1].lower() in depth_formats]
        ni, nv, nd = len(images), len(videos), len(depths)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos + depths
        self.nf = ni + nv + nd  # number of files
        self.video_flag = [0] * ni + [1] * nv + [2] * nd
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

        if annotations:
            with h5py.File(annotations, "r") as f:
                self.bounding_box = f["bounding_box"][:].copy()
                self.keypoints = f["image_coordinates"][:].copy()
            assert nd == self.bounding_box.shape[0], 'Files are not matched with annotations!'


    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        box = None
        keypoints = None

        if self.video_flag[self.count] == 2:
            # Read depth
            self.count += 1
            depth = np.load(path)  # BGR
            idx = int(Path(path).stem)
            box = self.bounding_box[idx]
            keypoints = self.keypoints[idx]
            assert depth is not None, 'Image Not Found ' + path
            img0 = D2RGB(depth)
            #print(f'image {self.count}/{self.nf} {path}: ', end='')
        else:
            if self.video_flag[self.count] == 1:
                # Read video
                self.mode = 'video'
                ret_val, img0 = self.cap.read()
                if not ret_val:
                    self.count += 1
                    self.cap.release()
                    if self.count == self.nf:  # last video
                        raise StopIteration
                    else:
                        path = self.files[self.count]
                        self.new_video(path)
                        ret_val, img0 = self.cap.read()

                self.frame += 1
                print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')
            else:
                # Read RGB image
                self.count += 1
                img0 = cv2.imread(path)  # BGR
                assert img0 is not None, 'Image Not Found ' + path
                #print(f'image {self.count}/{self.nf} {path}: ', end='')
            depth = RGB2D(img0)

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[np.newaxis, :, :, ::-1].transpose(0, 3, 1, 2) # BGR to RGB, to 1x3x416x416
        img = np.ascontiguousarray(img)

        img0 = img0[np.newaxis, ...]
        depth = depth[np.newaxis, ...]
        box = box[np.newaxis, ...]
        keypoints = keypoints[np.newaxis, ...]

        return path, img, img0, depth, box, keypoints, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


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


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            url = eval(s) if s.isnumeric() else s
            if 'youtube.com/' in str(url) or 'youtu.be/' in str(url):  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                url = pafy.new(url).getbest(preftype="mp4").url
            cap = cv2.VideoCapture(url)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) % 100

            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f' success ({w}x{h} at {self.fps:.2f} FPS).')
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(1 / self.fps)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years