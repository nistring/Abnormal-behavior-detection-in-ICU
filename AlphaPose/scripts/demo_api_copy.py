import os
import sys
from threading import Thread
from queue import Queue
import argparse
import platform
import time
from itertools import count
import asyncio
from concurrent.futures import ProcessPoolExecutor
from matplotlib import pyplot as plt

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

import torch
import torch.multiprocessing as mp

from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.pPose_nms import pose_nms, write_json

from action_recognition.action_recognition_api import ActionRecognition
from action_recognition.action_recognition_cfg import cfg as acfg

from tqdm import tqdm

from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track

from alphapose.utils.presets import SimpleTransform
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.vis import vis_frame_fast as vis_frame

import zmq
from EtherSense.EtherSenseClient import main, D2RGB, RGB2D

video_save_opt = {"savepath": "data/res/1.mp4", "fourcc": cv2.VideoWriter_fourcc(*"mp4v"), "fps": 15, "frameSize": (640, 480)}

EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description="AlphaPose Demo")
parser.add_argument("--cfg", type=str, required=True, help="experiment configure file name")
parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint file name")
parser.add_argument("--sp", default=False, action="store_true", help="Use single process for pytorch")
parser.add_argument("--detector", dest="detector", help="detector name", default="yolo")
parser.add_argument("--detfile", dest="detfile", help="detection result file", default="")
parser.add_argument("--list", dest="inputlist", help="image-list", default="")
parser.add_argument("--image", dest="inputimg", help="image-name", default="")
parser.add_argument("--outdir", dest="outputpath", help="output-directory", default="examples/res/")
parser.add_argument("--save_img", default=False, action="store_true", help="save result as image")
parser.add_argument("--vis", default=False, action="store_true", help="visualize image")
parser.add_argument("--showbox", default=False, action="store_true", help="visualize human bbox")
parser.add_argument("--profile", default=False, action="store_true", help="add speed profiling at screen output")
parser.add_argument("--format", type=str, help="save in the format of cmu or coco or openpose, option: coco/cmu/open")
parser.add_argument("--min_box_area", type=int, default=0, help="min box area to filter out")
parser.add_argument("--detbatch", type=int, default=3, help="detection batch size PER GPU")
parser.add_argument("--posebatch", type=int, default=3, help="pose estimation maximum number of people in an image")
parser.add_argument(
    "--eval",
    dest="eval",
    default=False,
    action="store_true",
    help="save the result json as coco format, using image index(int) instead of image name(str)",
)
parser.add_argument(
    "--gpus",
    type=str,
    dest="gpus",
    default="0",
    help="choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)",
)
parser.add_argument(
    "--qsize",
    type=int,
    dest="qsize",
    default=10,
    help="the length of result buffer, where reducing it will lower requirement of cpu memory",
)
parser.add_argument("--flip", default=False, action="store_true", help="enable flip testing")
parser.add_argument("--debug", default=False, action="store_true", help="print detail information")
"""----------------------------- Video options -----------------------------"""
parser.add_argument("--save_video", dest="save_video", help="whether to save rendered video", default=False, action="store_true")
parser.add_argument("--vis_fast", dest="vis_fast", help="use fast rendering", action="store_true", default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument("--pose_flow", dest="pose_flow", help="track humans in video with PoseFlow", action="store_true", default=False)
parser.add_argument("--pose_track", dest="pose_track", help="track humans in video with reid", action="store_true", default=False)
"""----------------------------- action recognition options -----------------------------"""
parser.add_argument("--action", dest="action", help="recognize action from skeleton", action="store_true", default=False)


args = parser.parse_args()
cfg = update_config(args.cfg)

if platform.system() == "Windows":
    args.sp = True

args.gpus = [int(i) for i in args.gpus.split(",")] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch  # * len(args.gpus)
args.posebatch = args.posebatch  # * len(args.gpus)
args.actionbatch = args.posebatch
args.tracking = args.pose_track or args.pose_flow or args.detector == "tracker"

if not args.sp:
    torch.multiprocessing.set_start_method("forkserver", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")


class DetectionLoader:
    def __init__(self, detector, cfg, opt, queueSize):
        self.cfg = cfg
        self.opt = opt
        self.device = opt.device

        self.detector = detector

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE

        self._sigma = cfg.DATA_PRESET.SIGMA

        if cfg.DATA_PRESET.TYPE == "simple":
            pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
            self.transformation = SimpleTransform(
                pose_dataset,
                scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0,
                sigma=self._sigma,
                train=False,
                add_dpg=False,
                gpu_device=self.device,
            )
        else:
            raise ValueError()

        # initialize the queue used to store data
        """
        image_queue: the buffer storing pre-processed images for object detection
        det_queue: the buffer storing human detection results
        pose_queue: the buffer storing post-processed cropped human image for pose estimation
        """
        if opt.sp:
            self._stopped = False
            self.image_queue = Queue(maxsize=queueSize)
            self.det_queue = Queue(maxsize=10 * queueSize)
            self.pose_queue = Queue(maxsize=10 * queueSize)
        else:
            self._stopped = mp.Value("b", False)
            self.image_queue = mp.Queue(maxsize=queueSize)
            self.det_queue = mp.Queue(maxsize=10 * queueSize)
            self.pose_queue = mp.Queue(maxsize=10 * queueSize)

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target, args=())
        # p.daemon = True
        p.start()
        return p

    def start(self):
        # start a thread to pre process images for object detection
        image_preprocess_worker = self.start_worker(self.image_preprocess)
        # start a thread to detect human in images
        image_detection_worker = self.start_worker(self.image_detection)
        # start a thread to post process cropped human image for pose estimation
        image_postprocess_worker = self.start_worker(self.image_postprocess)

        return [image_preprocess_worker, image_detection_worker, image_postprocess_worker]

    def stop(self):
        # clear queues
        self.clear_queues()

    def terminate(self):
        if self.opt.sp:
            self._stopped = True
        else:
            self._stopped.value = True
        self.stop()

    def clear_queues(self):
        self.clear(self.image_queue)
        self.clear(self.det_queue)
        self.clear(self.pose_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()

    def image_preprocess(self):
        ipc_queue = mp.Queue()
        stream = mp.Process(target=main, args=(ipc_queue,))
        stream.start()
        self.input_source = [data[0] for data in ipc_queue.get()]

        for i in count():
            imglist = [data[1] for data in ipc_queue.get()]
            imgs = []
            orig_imgs = []
            im_names = []
            im_dim_list = []
            for k in range(len(imglist)):
                if self.stopped:
                    self.wait_and_put(self.image_queue, (None, None, None, None))
                    self.stream.terminate()
                    self.stream.close()
                    return
                im_name_k = imglist[k]

                # expected image shape like (1,3,h,w) or (3,h,w)
                img_k = self.detector.image_preprocess(im_name_k)
                if isinstance(img_k, np.ndarray):
                    img_k = torch.from_numpy(img_k)
                # add one dimension at the front for batch if image shape (3,h,w)
                if img_k.dim() == 3:
                    img_k = img_k.unsqueeze(0)
                orig_img_k = cv2.cvtColor(im_name_k, cv2.COLOR_BGR2RGB)  # scipy.misc.imread(im_name_k, mode='RGB') is depreciated
                im_dim_list_k = orig_img_k.shape[1], orig_img_k.shape[0]

                imgs.append(img_k)
                orig_imgs.append(orig_img_k)
                im_names.append(str(i) + ".jpg")
                im_dim_list.append(im_dim_list_k)

            with torch.no_grad():
                # Human Detection
                imgs = torch.cat(imgs)
                im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
                # im_dim_list_ = im_dim_list

            self.wait_and_put(self.image_queue, (imgs, orig_imgs, im_names, im_dim_list))

    def image_detection(self):
        while True:
            imgs, orig_imgs, im_names, im_dim_list = self.wait_and_get(self.image_queue)
            if imgs is None or self.stopped:
                self.wait_and_put(self.det_queue, (None, None, None, None, None, None, None))
                self.stream.terminate()
                self.stream.close()
                return

            with torch.no_grad():
                dets = self.detector.images_detection(imgs, im_dim_list)
                if isinstance(dets, int) or dets.shape[0] == 0:
                    self.wait_and_put(self.det_queue, (orig_imgs, im_names, None, None, None, None, None, None))
                    continue
                if isinstance(dets, np.ndarray):
                    dets = torch.from_numpy(dets)
                dets = dets.cpu()

                # # Select boxes maximum number of posebatch each image then collate
                # indices = []
                # for k in range(len(orig_imgs)):
                #     idx = (dets[:, 0] == k).nonzero()[:,0]
                #     if idx.shape[0] >= self.opt.posebatch:
                #         idx = idx[:self.opt.posebatch]
                #     indices.append(idx)
                # indices = torch.cat(indices)
                # dets = dets[indices]
                batch_ids = dets[:, 0].long()
                boxes = dets[:, 1:5]
                scores = dets[:, 5:6]
                if self.opt.tracking:
                    ids = dets[:, 6:7]
                else:
                    ids = torch.zeros(scores.shape)

            if isinstance(boxes, int) or boxes.shape[0] == 0:
                self.wait_and_put(self.det_queue, (orig_imgs, im_names, None, None, None, None, None, None))
                continue
            inps = torch.zeros(boxes.size(0), 3, *self._input_size)
            cropped_boxes = torch.zeros(boxes.size(0), 4)

            self.wait_and_put(
                self.det_queue,
                (orig_imgs, im_names, boxes, scores, ids, inps, cropped_boxes, batch_ids),
            )

    def image_postprocess(self):
        while True:
            with torch.no_grad():
                (orig_imgs, im_names, boxes, scores, ids, inps, cropped_boxes, batch_ids) = self.wait_and_get(self.det_queue)
                if orig_imgs is None or self.stopped:
                    self.wait_and_put(self.pose_queue, (None, None, None, None, None, None, None, None))
                    self.stream.terminate()
                    self.stream.close()
                    return
                if boxes is None or boxes.nelement() == 0:
                    self.wait_and_put(self.pose_queue, (None, orig_imgs, im_names, boxes, scores, ids, None, None))
                    continue
                # imght = orig_img.shape[0]
                # imgwidth = orig_img.shape[1]

                for i, box in enumerate(boxes):

                    inps[i], cropped_box = self.transformation.test_transform(orig_imgs[batch_ids[i]], box)
                    cropped_boxes[i] = torch.FloatTensor(cropped_box)
                # inps, cropped_boxes = self.transformation.align_transform(orig_img, boxes)

                self.wait_and_put(self.pose_queue, (inps, orig_imgs, im_names, boxes, scores, ids, cropped_boxes, batch_ids))

    def read(self):
        return self.wait_and_get(self.pose_queue)

    @property
    def stopped(self):
        if self.opt.sp:
            return self._stopped
        else:
            return self._stopped.value


class DataWriter:
    def __init__(self, cfg, opt, save_video, queueSize, input_source, video_save_opt=None):
        self.cfg = cfg
        self.opt = opt
        self.video_save_opt = video_save_opt
        self.input_source = input_source

        self.eval_joints = list(range(cfg.DATA_PRESET.NUM_JOINTS))  # EVAL_JOINTS
        self.save_video = False  # save_video
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        # initialize the queue used to store frames read from
        # the video file
        if opt.sp:
            self.result_queue = Queue(maxsize=queueSize)
        else:
            self.result_queue = mp.Queue(maxsize=queueSize)

        if opt.save_img:
            for i in input_source:
                if not os.path.exists(opt.outputpath + f"/{i}"):
                    os.mkdir(opt.outputpath + f"/{i}")

        if opt.pose_flow:
            from trackers.PoseFlow.poseflow_infer import PoseFlowWrapper

            self.pose_flow_wrapper = PoseFlowWrapper(save_path=os.path.join(opt.outputpath, "poseflow"))

        if self.opt.save_img or self.save_video or self.opt.vis:
            loss_type = self.cfg.DATA_PRESET.get("LOSS_TYPE", "MSELoss")
            num_joints = self.cfg.DATA_PRESET.NUM_JOINTS
            if loss_type == "MSELoss":  # This is the loss
                self.vis_thres = [0.4] * num_joints
            elif "JointRegression" in loss_type:
                self.vis_thres = [0.05] * num_joints
            elif loss_type == "Combined":
                if num_joints == 68:
                    hand_face_num = 42
                else:
                    hand_face_num = 110
                self.vis_thres = [0.4] * (num_joints - hand_face_num) + [0.05] * hand_face_num

        self.use_heatmap_loss = self.cfg.DATA_PRESET.get("LOSS_TYPE", "MSELoss") == "MSELoss"

        self.action_model = ActionRecognition(acfg, self.opt, len(self.input_source)) if self.opt.action else None

        # for in_source in self.input_source:
        #     cv2.namedWindow(in_source, cv2.WINDOW_AUTOSIZE)
    # async def write_image(self, input_source, queue, canvas):
    #     while True:
    #         img, name, stream = await queue.get()
    #         if self.opt.vis:
    #             photo = ImageTk.PhotoImage(image=Image.fromarray(img))
    #             canvas.create_image(0, 0, image=photo, ancor=tk.NW)
    #         if self.opt.save_img:
    #             cv2.imwrite(os.path.join(self.opt.outputpath, input_source, name), img)
    #         if self.save_video:
    #             stream.write(img)
    #         if img is None:
    #             return

    # async def main(self, executor, queues):
    #     windows = []
    #     canvas = []
    #     for i_s in self.input_source:
    #         window = tk.Tk()
    #         window.title(i_s)
    #         c = tk.Canvas(window, width=640, height=480)
    #         c.pack()
    #         canvas.append(c)
    #         windows.append(window)

    #     await asyncio.get_event_loop().run_in_executor(
    #         executor, asyncio.gather((self.write_image(i_s, queues[i_s], c) for (i_s, c) in zip(self.input_source, canvas)))
    #     )

    #     for window in windows:
    #         window.mainloop()

    def start_worker(self, target):
        # queues = {}
        # for i_s in self.input_source:
        #     queues[i_s] = asyncio.Queue()
        if self.opt.sp:
            p = Thread(target=target, args=(self.action_model,))
        else:
            p = mp.Process(target=target, args=(self.action_model,))
        # p.daemon = True
        p.start()
        # excutor = ProcessPoolExecutor()
        # asyncio.get_event_loop().run_until_complete(self.main(excutor, queues))
        return p

    def start(self):
        # start a thread to read pose estimation results per frame
        self.result_worker = self.start_worker(self.update)
        return self

    def update(self, action_model):

        final_result = []
        norm_type = self.cfg.LOSS.get("NORM_TYPE", None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        if self.save_video:
            # initialize the file video stream, adapt ouput video resolution to original video
            stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ["savepath", "fourcc", "fps", "frameSize"]])
            if not stream.isOpened():
                print("Try to use other video encoders...")
                ext = self.video_save_opt["savepath"].split(".")[-1]
                fourcc, _ext = self.recognize_video_ext(ext)
                self.video_save_opt["fourcc"] = fourcc
                self.video_save_opt["savepath"] = self.video_save_opt["savepath"][:-4] + _ext
                stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ["savepath", "fourcc", "fps", "frameSize"]])
            assert stream.isOpened(), "Cannot open video for writing"
        # keep looping infinitelyd
        while True:
            # ensure the queue is not empty and get item
            (boxes_n, scores_n, ids_n, hm_data_n, cropped_boxes_n, orig_imgs, im_names, batch_ids) = self.wait_and_get(self.result_queue)
            if orig_imgs is None:
                # if the thread indicator variable is set (img is None), stop the thread
                if self.save_video:
                    stream.release()
                # write_json(final_result, self.opt.outputpath, form=self.opt.format, for_eval=self.opt.eval)
                print("Results have been written to json.")
                return
            # image channel RGB->BGR
            rets = []
            # print(boxes_n, scores_n, ids_n, hm_data_n, cropped_boxes_n, orig_imgs, im_names, batch_ids)
            for i, (orig_img, im_name) in enumerate(zip(orig_imgs, im_names)):
                # This operation still preserves dimension

                batch_id = batch_ids == i
                orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
                if batch_id is False or torch.all(batch_id == False):
                    if self.opt.save_img or self.save_video or self.opt.vis:
                        self.write_image(orig_img, im_name, stream if self.save_video else None, self.input_source[i])
                        #queues[self.input_source[i]].put(orig_img, im_name, stream if self.save_video else None)
                    rets.append(None)
                else:
                    pass
                    boxes = boxes_n[batch_id]
                    scores = scores_n[batch_id]
                    ids = ids_n[batch_id]
                    hm_data = hm_data_n[batch_id]
                    cropped_boxes = cropped_boxes_n[batch_id]
                    # location prediction (n, kp, 2) | score prediction (n, kp, 1)
                    assert hm_data.dim() == 4

                    pose_coords = []
                    pose_scores = []

                    for i in range(hm_data.shape[0]):
                        bbox = cropped_boxes[i].tolist()
                        pose_coord, pose_score = self.heatmap_to_coord(
                            hm_data[i][self.eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type
                        )
                        pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                        pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))

                    preds_img = torch.cat(pose_coords)
                    preds_scores = torch.cat(pose_scores)

                    # boxes, scores, ids, preds_img, preds_scores, pick_ids
                    rets.append(
                        pose_nms(
                            boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area, use_heatmap_loss=self.use_heatmap_loss
                        )
                    )

            if self.opt.action:
                _, anomaly_scores = action_model(rets)

            for i, ret in enumerate(rets):
                if ret:
                    boxes, scores, ids, preds_img, preds_scores, pick_ids = ret
                    _result = []
                    for k in range(len(scores)):
                        _result.append(
                            {
                                "keypoints": preds_img[k],
                                "kp_score": preds_scores[k],
                                "proposal_score": torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                                "idx": ids[k],
                                "box": [boxes[k][0], boxes[k][1], boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1]],
                                "anomaly_score": anomaly_scores[i] if self.opt.action and anomaly_scores else 0,
                            }
                        )

                    result = {"imgname": im_name, "result": _result}

                    final_result.append(result)
                    if self.opt.save_img or self.save_video or self.opt.vis:
                        img = vis_frame(orig_img, result, self.opt, self.vis_thres)
                        self.write_image(img, im_name, stream if self.save_video else None, self.input_source[i])
                        #queues[self.input_source[i]].put(img, im_name, stream if self.save_video else None)

    def write_image(self, img, name, stream, input_source):
        if self.opt.vis:
            cv2.imshow(input_source, img)
            k = cv2.waitKey(1)
        if self.opt.save_img:
            cv2.imwrite(os.path.join(self.opt.outputpath, input_source, name), img)
        if self.save_video:
            stream.write(img)

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()

    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name, batch_ids):
        # save next frame in the queue
        self.wait_and_put(self.result_queue, (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name, batch_ids))

    def running(self):
        # indicate that the thread is still running
        return not self.result_queue.empty()

    def count(self):
        # indicate the remaining images
        return self.result_queue.qsize()

    def stop(self):
        # indicate that the thread should be stopped
        self.save(None, None, None, None, None, None, None, None)
        self.result_worker.join()

    def terminate(self):
        # directly terminate
        self.result_worker.terminate()

    def clear_queues(self):
        self.clear(self.result_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def results(self):
        # return final result
        print(self.final_result)
        return self.final_result

    def recognize_video_ext(self, ext=""):
        if ext == "mp4":
            return cv2.VideoWriter_fourcc(*"mp4v"), "." + ext
        elif ext == "avi":
            return cv2.VideoWriter_fourcc(*"XVID"), "." + ext
        elif ext == "mov":
            return cv2.VideoWriter_fourcc(*"XVID"), "." + ext
        else:
            print("Unknow video format {}, will use .mp4 instead of it".format(ext))
            return cv2.VideoWriter_fourcc(*"mp4v"), ".mp4"


def print_finish_info():
    print("===========================> Finish Model Running.")
    if (args.save_img or args.save_video) and not args.vis_fast:
        print("===========================> Rendering remaining images in the queue...")
        print(
            "===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time)."
        )


def loop():
    n = 0
    while True:
        yield n
        n += 1


if __name__ == "__main__":
    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    # Load detection loader
    det_loader = DetectionLoader(get_detector(args), cfg, args, queueSize=args.qsize)
    det_worker = det_loader.start()
    input_source = ['192.168.0.4:1024', '192.168.0.4:1025', '192.168.0.4:1026']

    # Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print("Loading pose model from %s..." % (args.checkpoint,))
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    if args.pose_track:
        tracker = Tracker(tcfg, args)
    if len(args.gpus) > 1:
        pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
    else:
        pose_model.to(args.device)
    pose_model.eval()

    runtime_profile = {"dt": [], "pt": [], "pn": []}

    # Init data writer
    queueSize = 2
    if args.save_video:
        video_save_opt["savepath"] = [os.path.join(args.outputpath, "AlphaPose_webcam" + str(i) + ".mp4") for i in input_source]
        writer = DataWriter(
            cfg, args, save_video=True, video_save_opt=video_save_opt, queueSize=queueSize, input_source=input_source
        ).start()
    else:
        writer = DataWriter(cfg, args, save_video=False, queueSize=queueSize, input_source=input_source).start()

    print("Starting webcam demo, press Ctrl + C to terminate...")
    sys.stdout.flush()
    im_names_desc = tqdm(loop())

    batchSize = args.posebatch
    if args.flip:
        batchSize = int(batchSize / 2)
    try:
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_imgs, im_names, boxes, scores, ids, cropped_boxes, batch_ids) = det_loader.read()
                if orig_imgs is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_imgs, im_names, batch_ids)
                    continue
                if args.profile:
                    ckpt_time, det_time = getTime(start_time)
                    runtime_profile["dt"].append(det_time)

                # Pose Estimation
                inps = inps.to(args.device)
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize : min((j + 1) * batchSize, datalen)]
                    if args.flip:
                        inps_j = torch.cat((inps_j, flip(inps_j)))
                    hm_j = pose_model(inps_j)
                    if args.flip:
                        hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2) :], pose_dataset.joint_pairs, shift=True)
                        hm_j = (hm_j[0 : int(len(hm_j) / 2)] + hm_j_flip) / 2
                    hm.append(hm_j)
                hm = torch.cat(hm)

                if args.profile:
                    ckpt_time, pose_time = getTime(ckpt_time)
                    runtime_profile["pt"].append(pose_time)
                if args.pose_track:
                    raise ValueError("Pose track is not available")
                    boxes, scores, ids, hm, cropped_boxes = track(tracker, args, orig_img, inps, boxes, hm, cropped_boxes, im_name, scores)

                hm = hm.cpu()
                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_imgs, im_names, batch_ids)
                if args.profile:
                    ckpt_time, post_time = getTime(ckpt_time)
                    runtime_profile["pn"].append(post_time)

            if args.profile:
                # TQDM
                im_names_desc.set_description(
                    "det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}".format(
                        dt=np.mean(runtime_profile["dt"]), pt=np.mean(runtime_profile["pt"]), pn=np.mean(runtime_profile["pn"])
                    )
                )
        print_finish_info()
        while writer.running():
            time.sleep(1)
            print("===========================> Rendering remaining " + str(writer.count()) + " images in the queue...", end="\r")
        writer.stop()
        det_loader.stop()

    except Exception as e:
        print(repr(e))
        print("An error as above occurs when processing the images, please check it")
        pass
    except KeyboardInterrupt:
        print_finish_info()
        # Thread won't be killed when press Ctrl+C
        if args.sp:
            det_loader.terminate()
            while writer.running():
                time.sleep(1)
                print("===========================> Rendering remaining " + str(writer.count()) + " images in the queue...", end="\r")
            writer.stop()
        else:
            # subprocesses are killed, manually clear queues

            det_loader.terminate()
            writer.terminate()
            writer.clear_queues()
            det_loader.clear_queues()
