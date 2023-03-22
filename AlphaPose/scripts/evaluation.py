from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import tempfile
import json
from sklearn.metrics import roc_curve, RocCurveDisplay, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

if __name__ == '__main__':
    anno_json = 'data/coco/annotations/person_keypoints_test.json'
    pred_json = 'data/res2/alphapose-results.json'

    keypoints = [
        "Nose", "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
        "Left Wrist", "Right Wrist", "Left Hip", "Right Hip"
    ]

    with open(anno_json) as f:
        anno_dict = json.load(f)
    with open(pred_json) as f:
        pred_dict = json.load(f)
    for anno in pred_dict:
        anno['bbox'] = anno.pop('box')
    _, tmp = tempfile.mkstemp()
    json.dump(pred_dict, open(tmp, "w"))

    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(tmp)  # init predictions api

    # Keypoints
    eval = COCOeval(anno, pred, "keypoints")
    eval.params.kpt_oks_sigmas = np.array([.26, .79, .79, .72, .72, .62,.62, 1.07, 1.07])/10.0
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    # Bbox
    eval = COCOeval(anno, pred, "bbox")
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    # Action
    seen = []
    action_test = []
    for anno in anno_dict['annotations']:
        if anno['id'] not in seen:
            seen.append(anno['id'])
            action_test.append(anno['normal_score'])
    action_score = [pred['normal_score'] for pred in pred_dict]

    assert len(action_test) == len(action_score)

    valid_frames = [idx for idx, score in enumerate(action_test) if score != None]
    action_test = [1-action_test[i] for i in valid_frames]
    action_score = [1-action_score[i] for i in valid_frames]

    fpr, tpr, _ = roc_curve(action_test, action_score)
    auc = roc_auc_score(action_test, action_score)
    print(f"Action recognition AUC : {auc}")

    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.show()

    action_pred = [1 if score > 0.5 else 0 for score in action_score]
    cm = confusion_matrix(action_test, action_pred)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.show()