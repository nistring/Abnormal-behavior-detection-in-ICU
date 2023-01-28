set -x

python test.py \
    --dt_weights yolov7/best.pt \
    --pe_weights A2J/bestHPE.pt\
    --save-json \
    --classes 0 \
    --source /media/nistring/0f737a3f-358d-45e0-827c-473d6a7d555b/workspace/Abnormal-behavior-detection-in-ICU/datasets/test/depth_data \
    --nosave \
    --annotations /media/nistring/0f737a3f-358d-45e0-827c-473d6a7d555b/workspace/Abnormal-behavior-detection-in-ICU/datasets/test/ICU_test_labels.h5 \
    --exist-ok
    # --webcam 3 \