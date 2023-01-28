set -x

# CONFIG=$1
# CKPT=$2
# VIDEO=$3
# OUTDIR=${4:-"./examples/res"}

# python scripts/demo_inference.py \
#     --cfg ${CONFIG} \
#     --checkpoint ${CKPT} \
#     --video ${VIDEO} \
#     --outdir ${OUTDIR} \
#     --detector yolo  --save_img --save_video

CONFIG=${1:-"configs/coco/resnet/256x192_res50_lr1e-3_2x-dcn.yaml"}
CKPT=${2:-"pretrained_models/fast_dcn_res50_256x192.pth"}
# INDIR=${3:-"/home/ubuntu/workspace/@ICU/AlphaPose/examples/demo"}
INDIR=${3:-"../datasets/test/color_img"}
OUTDIR=${4:-"../datasets/test/res"}

python scripts/demo_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --indir ${INDIR} \
    --outdir ${OUTDIR} \
    --dtype test --detbatch 8 --posebatch 64 \
    # --save_img  --showbox