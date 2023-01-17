set -x

python test.py \
    --dt_weights yolov7/best.pt \
    --pe_weights A2J/bestHPE.pt\
    --save-json \
    --classes 0 \
    --webcam 3 \
    --nosave \
    --view-img