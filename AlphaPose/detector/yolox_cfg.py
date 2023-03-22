from easydict import EasyDict as edict

cfg = edict()
cfg.MODEL_NAME = ""#"yolox-m"
cfg.MODEL_WEIGHTS = ""#"detector/YOLOX/data/yolox_m_icu.pth"
cfg.INP_DIM = 640
cfg.CONF_THRES = 0.1
cfg.NMS_THRES = 0.0
