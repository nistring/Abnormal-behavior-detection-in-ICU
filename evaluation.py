from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

anno_json = '/media/nistring/0f737a3f-358d-45e0-827c-473d6a7d555b/workspace/Abnormal-behavior-detection-in-ICU/runs/exp/annotations.json'
pred_json = '/media/nistring/0f737a3f-358d-45e0-827c-473d6a7d555b/workspace/Abnormal-behavior-detection-in-ICU/runs/exp/best_bestHPE_predictions.json'

keypoints = [
    "Nose", "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip"
]

anno = COCO(anno_json)  # init annotations api
pred = anno.loadRes(pred_json)  # init predictions api
eval = COCOeval(anno, pred, "keypoints")
eval.params.kpt_oks_sigmas = np.array([.26, .79, .79, .72, .72, .62,.62, 1.07, 1.07])/10.0
eval.evaluate()
eval.accumulate()
eval.summarize()