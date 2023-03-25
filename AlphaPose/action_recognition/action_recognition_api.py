# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa: E722
import os.path as osp
import torch
from operator import itemgetter

from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint

import os, sys
root = os.path.dirname(__file__)
sys.path.append(root)

from pyskl.models import build_model
from pyskl.utils import cache_checkpoint
from pyskl.datasets.pipelines import Compose

from collections import deque
import numpy as np
from multiprocessing import Queue, Process

class ActionRecognition():
    def __init__(self, acfg, opt, batch_size=1):
        self.acfg = acfg
        self.opt = opt
        self.idx = 0
        self.ret = (None, None)
        self.in_queue = Queue(1)
        self.out_queue = Queue(1)
        self.batch_size = batch_size
        
        self.model = self.init_recognizer()
        Process(target=self.action_model, args=()).start()

    def init_recognizer(self):
        cfg = Config.fromfile(self.acfg.config)

        self.pipeline = cfg.test_pipeline
        self.pipeline[0]['num_clips'] = 1
        self.pipeline[-2]['keys'] = ['imgs']
        self.clip_len = self.pipeline[0]['clip_len']
        self.queue = []
        for i in range(self.batch_size):
            self.queue.append(deque([], maxlen=self.clip_len))
        self.pipeline = Compose(self.pipeline)

        # set cudnn benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.data.test.test_mode = True

        """Get predictions by pytorch models."""
        if self.acfg.average_clips is not None:
            # You can set average_clips during testing, it will override the
            # original setting
            if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
                cfg.model.setdefault('test_cfg',
                                    dict(average_clips=self.acfg.average_clips))
            else:
                if cfg.model.get('test_cfg') is not None:
                    cfg.model.test_cfg.average_clips = self.acfg.average_clips
                else:
                    cfg.test_cfg.average_clips = self.acfg.average_clips

        # build the model and load checkpoint
        model = build_model(cfg.model)

        if self.acfg.checkpoint is None:
            work_dir = cfg.work_dir
            self.acfg.checkpoint = osp.join(work_dir, 'latest.pth')
            assert osp.exists(self.acfg.checkpoint)

        self.acfg.checkpoint = cache_checkpoint(self.acfg.checkpoint)
        import os
        import contextlib

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            load_checkpoint(model, self.acfg.checkpoint, map_location='cpu')
        print(f"Loading action recognition model from {self.acfg.checkpoint}")

        if self.acfg.fuse_conv_bn:
            model = fuse_conv_bn(model)

        if len(self.opt.gpus) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.opt.gpus).to(self.opt.device)
        else:
            model.to(self.opt.device)
        model.eval()

        return model

    def action_model(self):
        kps = self.in_queue.get()
        inps = []
        for i in range(kps.shape[0]):
            results = {
                'total_frames' : self.clip_len,
                'img_shape' : (480,640),
                'original_shape' : (480,640),
                'keypoint' : kps[i:i+1,:,:,:-1],
                'keypoint_score' : kps[i:i+1,:,:,-1],
            }
            inps.append(self.pipeline(results)['imgs'])
        inps = torch.stack(inps).to(self.opt.device)
        with torch.no_grad():
            scores = self.model(inps, return_loss=False)
        top_1 = np.argmax(scores, axis=1)
        self.out_queue.put((top_1, scores[:, 1]))

    def __call__(self, rets):
        for i, ret in enumerate(rets):
            if ret:
                boxes, scores, ids, preds_img, preds_scores, pick_ids = ret
                if len(boxes) == 1:
                    self.queue[i].append(torch.cat((preds_img[0], preds_scores[0]), dim=1).numpy())

        for q in self.queue:
            if len(q) != self.clip_len:
                return self.ret
        
        if self.in_queue.empty():
            self.in_queue.put(np.stack([np.array(q) for q in self.queue]))
            
        if not self.out_queue.empty():
            self.ret = self.out_queue.get()
        
        return self.ret