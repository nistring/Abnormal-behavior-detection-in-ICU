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

CONFIG=${1:-"configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"}
CKPT=${2:-"pretrained_models/fast_res50_256x192.pth"}
INDIR=${3:-"../data/train/color_img"}
OUTDIR=${4:-"../data/train/res"}

python scripts/demo_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --indir ${INDIR} \
    --outdir ${OUTDIR} \
    --detbatch 4 --posebatch 32 --eval\

INDIR=${3:-"../data/test/color_img"}
OUTDIR=${4:-"../data/test/res"}

python scripts/demo_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --indir ${INDIR} \
    --outdir ${OUTDIR} \
    --detbatch 4 --posebatch 32 --eval\

INDIR=${3:-"../data/val/color_img"}
OUTDIR=${4:-"../data/val/res"}

python scripts/demo_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --indir ${INDIR} \
    --outdir ${OUTDIR} \
    --detbatch 4 --posebatch 32 --eval\