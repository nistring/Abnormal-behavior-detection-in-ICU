from easydict import EasyDict as edict
import os
cfg = edict()
cfg.config = "action_recognition/configs/posec3d/x3d_shallow_icu/limb.py"
cfg.checkpoint = "action_recognition/data/x3d_icu.pth"
cfg.fuse_conv_bn = False
cfg.eval = ['top_k_accuracy', 'mean_class_accuracy']
cfg.tmpdir = None
cfg.average_clips = None # ['score', 'prob', None]
cfg.launcher = 'pytorch'
cfg.local_rank = 0

if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = str(cfg.local_rank)