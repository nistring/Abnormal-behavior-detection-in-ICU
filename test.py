import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os
import torch.utils.data
import numpy as np
import yaml
import json
import sys
import paramiko

root = os.path.abspath(os.path.dirname(__file__))
res_dir = os.path.join(root, "res")

sys.path.append(os.path.join(root, "yolov7"))
sys.path.append(os.path.join(root, "A2J"))
sys.path.append(os.path.join(root, "A2J", "src"))

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import (
    select_device,
    load_classifier,
    time_synchronized,
    TracedModel,
)

import A2J.src.model
from A2J.src.dataloader import build_test_dataloader

from loader import LoadImages, LoadWebcam


def make_coco_dataset():
    skeleton = [
        [0,0],[1,2],[1,3],[2,4],[3,5],[4,6],[7,8],[2,8],[1,7]
    ]

    coco = {

        "info": {
            "description": "",
            "url": "http://bilab.ai/",
            "version": "1.0",
            "year": 2022,
            "contributor": "",
            "data_created": "2022/07/09"
        },

        "licenses": [
            # For example
            {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
            }
        ],

        "images": [
            # ***Format***
            # {
            #     "license": 1,
            #     "file_name": "000000397133.jpg",
            #     "height": 480,
            #     "width": 640,
            #     "date_captured": "2020-08-31T12:43:47.223Z"
            #     "flickr_url": None,
            #     "classId": 1496904
            #     "id": 506412614
        ],

        "categories": [
            {
                "supercategory": "person",
                "id": 0,
                "name": "patient",
                "keypoints": [
                    "Nose", "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
                    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
                ],
                "skeleton": skeleton
            }
        ],

        "annotations": [
            # ***Format***
            # {
            #     "segmentation": None,
            #     "num_keypoints": 29,
            #     "area": None,
            #     "iscrowd": None,
            #     "keypoints": [],
            #     "image_id": 506412614,
            #     "bbox": [],
            #     "category_id": int,
            #     "id": None
            # }
        ]

    }
    return coco

def resize_keypoints(keypoints, box, input_size):
    return keypoints * (box[2:] - box[:2]) / input_size + box[:2]


def plot_skeleton(frame, keypoints, linewidth=3):
    color = (255, 255, 255)
    p = keypoints.numpy().astype(np.int16)
    cv2.line(
        frame,
        p[0],
        (p[1] + p[2]) // 2,
        color,
        linewidth,
    )
    cv2.line(frame, p[1], p[2], color, linewidth)
    cv2.line(frame, p[1], p[3], color, linewidth)
    # cv2.line(frame, p[1], p[7], (102,255,178), linewidth)
    cv2.line(frame, p[2], p[4], color, linewidth)
    # cv2.line(frame, p[2], p[8], (255,178,102), linewidth)
    cv2.line(frame, p[3], p[5], color, linewidth)
    cv2.line(frame, p[4], p[6], color, linewidth)
    cv2.line(frame, p[7], p[8], color, linewidth)
    cv2.line(frame, p[2], p[8], color, linewidth)
    cv2.line(frame, p[1], p[7], color, linewidth)


def test(save_img=False):
    (source, annotations, webcam, dt_weights, pe_weights, view_img, save_json, imgsz, trace,) = (
        opt.source,
        opt.annotations,
        opt.webcam,
        opt.dt_weights,
        opt.pe_weights,
        opt.view_img,
        opt.save_json,
        opt.img_size,
        not opt.no_trace,
    )
    with open(opt.config) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        input_size = cfg["input size"]
    save_img = not opt.nosave  # save inference images

    if source and webcam:
        raise ValueError("can't handle multiple sources!")

    device = select_device(opt.device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(dt_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Load pose estimation model
    net = A2J.src.model.A2J_model(num_classes=cfg["kp num"])
    net.load_state_dict(torch.load(pe_weights, map_location=device))
    net = net.to(device)
    net.eval()

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadWebcam(webcam, img_size=imgsz, stride=stride)
        for i in range(webcam):
            cv2.namedWindow(str(i))

    elif source:
        dataset = LoadImages(source, annotations, img_size=imgsz, stride=stride)
    else:
        raise AttributeError("no source or webcam is assigned")

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    if save_img:
        if webcam:
            for w in range(webcam):
                (save_dir / "images" / w).mkdir(parents=True, exist_ok=True)  # make dir
        else:
            (save_dir / "images").mkdir(parents=True, exist_ok=True)  # make dir
    else:
        (save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    jdict = []
    gt_coco = make_coco_dataset()

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != "cpu":
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, depth_map, depth, gt_box, gt_keypoints, vid_cap in dataset:
        path = Path(path)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # Warmup
        if device.type != "cpu" and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms,
        )
        t3 = time_synchronized()

        # Crop image
        cropped_img = []
        boxes = []
        for i, det in enumerate(pred):
            if len(det):
                int_xyxy = np.array(det[0, :4].cpu(), dtype=np.int16)
                boxes.append(int_xyxy)
                cropped_img.append(
                    cv2.resize(
                        depth[i, max(int_xyxy[1], 0) : min(int_xyxy[3], 480), max(int_xyxy[0], 0) : min(int_xyxy[2], 640)],
                        (input_size, input_size),
                    )
                )

        if len(cropped_img):
            cropped_img = torch.from_numpy(np.stack(cropped_img)).to(device)
            cropped_img = (cropped_img - cropped_img.mean()) / cropped_img.std()

            # Pose estimation
            dt_keypoints = net(cropped_img).cpu()

            kp_idx = 0
        t4 = time_synchronized()
        # Process detections
        for i, det in enumerate(pred):  # detections per image

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], depth_map[i].shape)

                # Write results
                #  Only use obj with highest confidence
                *xyxy, conf, cls = det[0]
                # for *xyxy, conf, cls in reversed(det):
                dt_keypoint = resize_keypoints(dt_keypoints[kp_idx], boxes[kp_idx], input_size)

                if save_json and source:
                    # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                    box = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1)  # xywh
                    box[:2] -= box[2:] / 2  # xy center to top-left corner
                    box = box.tolist()
                    conf = conf.item()

                    num_keypoints = dt_keypoint.shape[0]
                    visibility = np.ones((num_keypoints, 1)) * 2

                    jdict.append(
                        {
                            "area": box[2]*box[3],
                            "iscrowd": 0,
                            "id": image_id,
                            "image_id": image_id,
                            "category_id": int(cls),
                            "num_keypoints": num_keypoints,
                            "keypoints": np.concatenate((dt_keypoint, visibility), axis=1).reshape(-1).tolist(),
                            "bbox": [round(x, 3) for x in box],
                            "score": round(conf, 5),
                        }
                    )

                    # Write ground truth json
                    if annotations:
                        box = gt_box[i]
                        box[:2] -= box[2:] / 2  # xy center to top-left corner
                        box[[0,2]] *= 640
                        box[[1,3]] *= 480
                        box = box.tolist()

                        gt_keypoint = gt_keypoints[i]
                        num_keypoints = gt_keypoint.shape[0]
                        visibility = np.ones((num_keypoints, 1)) * 2
                        gt_coco["annotations"].append(
                            {
                                "area": box[2]*box[3],
                                "iscrowd": 0,
                                "id": image_id,
                                "image_id": image_id,
                                "category_id": 0,
                                "num_keypoints": num_keypoints,
                                "keypoints": np.concatenate((gt_keypoint, visibility), axis=1).reshape(-1).tolist(),
                                "bbox": [round(x, 3) for x in box],
                            }
                        )
                        gt_coco["images"].append(
                            {
                                "license": 1,
                                "file_name": path.name,
                                "height": 480,
                                "width": 640,
                                "id": image_id,
                            }
                        )

                if save_img or view_img:  # Add bbox to image
                    label = f"{names[int(cls)]} {conf:.2f}"
                    plot_one_box(
                        xyxy,
                        depth_map[i],
                        label=label,
                        color=colors[int(cls)],
                        line_thickness=1,
                    )
                    plot_skeleton(depth_map[i], dt_keypoint)  # torch.Tensor(gt_keypoint))

                kp_idx += 1

            # Print time (inference + NMS)
            print(f"({(1E3 * (t2 - t1)):.1f}ms) Detection, ({(1E3 * (t3 - t2)):.1f}ms) NMS, ({(1E3 * (t4 - t3)):.1f}ms) Pose Estimation")

            # Stream results
            if view_img:
                for i in range(webcam):
                    bigDepth = cv2.resize(depth_map[i], (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                    cv2.imshow(str(i), cv2.cvtColor(bigDepth, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

            # Save results (image with detections)
            if save_img:
                if webcam:
                    for i in range(webcam):
                        cv2.imwrite(
                            os.path.join(save_dir, "images", i, (path.stem + ".jpg")),
                            depth_map[i],
                        )
                else:
                    cv2.imwrite(
                        os.path.join(save_dir, "images", (path.stem + ".jpg")),
                        depth_map[i],
                    )
                print(f" The image with the result is saved in: {save_dir}")

    # Save JSON
    if save_json and len(jdict):
        dt_w = Path(dt_weights[0] if isinstance(dt_weights, list) else dt_weights).stem if dt_weights is not None else ""  # dt_weights
        pe_w = Path(pe_weights[0] if isinstance(pe_weights, list) else pe_weights).stem if pe_weights is not None else ""  # pe_weights
        anno_json = str(save_dir / "annotations.json")  # annotations json
        pred_json = str(save_dir / f"{dt_w}_{pe_w}_predictions.json")  # predictions json
        print("\nEvaluating pycocotools mAP... saving %s..." % pred_json)
        with open(pred_json, "w") as f:
            json.dump(jdict, f)
        if annotations:
            with open(anno_json, "w") as f:
                json.dump(gt_coco, f)

    print(f"Done. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    # server = ['192.168.0.4', '192.168.0.6', '192.168.0.7']
    # client = []
    # for i in range(3):
    #     cli = paramiko.SSHClient()
    #     cli.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    #     ser = server[i]
    #     cli.connect(ser, port=22, username='', password='')
    #     cli.exec_command('conda activate base')
    #     stdin, stdout, stderr = cli.exec_command('python /Users/cnu_r/OneDrive/Desktop/EtherSense/EtherSenseServer.py')
    #     client.append(cli)
    # try:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dt_weights",
        nargs="+",
        type=str,
        default="yolov7.pt",
        help="object detection model.pt path(s)",
    )
    parser.add_argument(
        "--pe_weights",
        type=str,
        default="",
        help="pose estimation model.pt path(s)",
    )
    parser.add_argument("--source", type=str, help="source")
    parser.add_argument("--webcam", type=int, help="input camera numbers")
    parser.add_argument("--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="object confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-json", action="store_true", help="save results to *.json")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default="runs", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument("--no-trace", action="store_true", help="don`t trace model")

    parser.add_argument("--annotations", type=str, help="path to annotations file")
    parser.add_argument("--config", type=str, default="A2J/src/cfg.yaml", help="path to configuration")
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
        test()

        # for c in client:
        #     c.close()
