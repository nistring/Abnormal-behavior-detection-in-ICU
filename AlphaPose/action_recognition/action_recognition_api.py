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

class ActionRecognition():
    def __init__(self, acfg, args):
        self.acfg = acfg
        self.args = args
        
        self.model = self.init_recognizer()

    def init_recognizer(self):
        cfg = Config.fromfile(self.acfg.config)

        self.pipeline = cfg.test_pipeline
        self.pipeline[0]['num_clips'] = 1
        self.pipeline[-2]['keys'] = ['imgs']
        self.clip_len = self.pipeline[0]['clip_len']
        self.queue = deque([], maxlen=self.clip_len)
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

        if len(self.args.gpus) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.args.gpus).to(self.args.device)
        else:
            model.to(self.args.device)
        model.eval()

        return model

    def __call__(self, preds_img, preds_score):
        if preds_img and preds_score:
            input = torch.cat((preds_img[0], preds_score[0]), dim=1).numpy()
            self.queue.append(input)
        if len(self.queue) == self.clip_len:
            # forward the model
            kp = np.array(self.queue)
            results = {
                'total_frames' : self.clip_len,
                'img_shape' : (480,640),
                'original_shape' : (480,640),
                'keypoint' : kp[np.newaxis,:,:,:-1],
                'keypoint_score' : kp[np.newaxis,:,:,-1],
            }
            results = self.pipeline(results)
            results = torch.unsqueeze(results['imgs'], 0).to(self.args.device)
            with torch.no_grad():
                scores = self.model(results, return_loss=False)[0]
            top_1 = np.argmax(scores)
            return (int(top_1), scores[1])

        return (None, None)