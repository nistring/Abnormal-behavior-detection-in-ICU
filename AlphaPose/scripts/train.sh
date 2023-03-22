set -x

CONFIG=${1:-'./configs/coco/custom/256x192_res50_lr1e-3_1x.yaml'}
EXPID=${2:-"alphapose"}

python ./scripts/train.py \
    --exp-id ${EXPID} \
    --cfg ${CONFIG} \
    --nThreads 16 --map --snapshot 2