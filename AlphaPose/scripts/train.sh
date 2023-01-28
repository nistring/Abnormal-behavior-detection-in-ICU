set -x

CONFIG=${1:-'/media/nistring/0f737a3f-358d-45e0-827c-473d6a7d555b/workspace/Abnormal-behavior-detection-in-ICU/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x-simple.yaml'}
EXPID=${2:-"alphapose"}

python ./scripts/train.py \
    --exp-id ${EXPID} \
    --cfg ${CONFIG} \
    --nThreads 32

CONFIG=${1:-'/media/nistring/0f737a3f-358d-45e0-827c-473d6a7d555b/workspace/Abnormal-behavior-detection-in-ICU/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml'}

python ./scripts/train.py \
    --exp-id ${EXPID} \
    --cfg ${CONFIG} \
    --nThreads 32

CONFIG=${1:-'/media/nistring/0f737a3f-358d-45e0-827c-473d6a7d555b/workspace/Abnormal-behavior-detection-in-ICU/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x-A2J.yaml'}

python ./scripts/train.py \
    --exp-id ${EXPID} \
    --cfg ${CONFIG} \
    --nThreads 32