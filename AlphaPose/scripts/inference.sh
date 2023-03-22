set -x

# generating pseudolabel

# CONFIG=${1:-"configs/coco/custom/256x192_res50_lr1e-3_1x_pretrained.yaml"}
# CKPT=${2:-"pretrained_models/fast_res50_256x192.pth"}
# INDIR=${3:-"../data/train/color_img"}
# OUTDIR=${4:-"../data/train/res"}

# python scripts/demo_inference.py \
#     --cfg ${CONFIG} \
#     --checkpoint ${CKPT} \
#     --indir ${INDIR} \
#     --outdir ${OUTDIR} \
#     --detector yolox-l  --eval

# CONFIG=${1:-"configs/coco/custom/256x192_res50_lr1e-3_1x_pretrained.yaml"}
# CKPT=${2:-"pretrained_models/fast_res50_256x192.pth"}
# INDIR=${3:-"../data/val/color_img"}
# OUTDIR=${4:-"../data/val/res"}

# python scripts/demo_inference.py \
#     --cfg ${CONFIG} \
#     --checkpoint ${CKPT} \
#     --indir ${INDIR} \
#     --outdir ${OUTDIR} \
#     --detector yolox-l  --eval

# CONFIG=${1:-"configs/coco/custom/256x192_res50_lr1e-3_1x_pretrained.yaml"}
# CKPT=${2:-"pretrained_models/fast_res50_256x192.pth"}
# INDIR=${3:-"../data/test/color_img"}
# OUTDIR=${4:-"../data/test/res"}

# python scripts/demo_inference.py \
#     --cfg ${CONFIG} \
#     --checkpoint ${CKPT} \
#     --indir ${INDIR} \
#     --outdir ${OUTDIR} \
#     --detector yolox-l  --eval

# Inference

CONFIG=${1:-"configs/coco/custom/256x192_res50_lr1e-3_1x.yaml"}
CKPT=${2:-"pretrained_models/fast_res50_256x192_icu.pth"}
INDIR=${3:-"data/coco/test"}
OUTDIR=${4:-"data/res2"}

python scripts/demo_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --indir ${INDIR} \
    --outdir ${OUTDIR} \
    --detector yolox-l-icu \
    --detbatch 4 --posebatch 32 --eval --action # --save_video --vis_fast --save_img --showbox\