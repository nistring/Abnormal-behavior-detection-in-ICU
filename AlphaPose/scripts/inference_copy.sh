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
OUTDIR=${3:-"data/res3"}

python scripts/demo_api_copy.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --outdir ${OUTDIR} \
    --detector yolox-l-icu \
    --detbatch 3 --posebatch 3 --action --eval --profile #--vis #  --vis # --save_video --vis_fast --save_img --showbox\

#CONFIG=${1:-"configs/coco/custom/256x192_res50_lr1e-3_1x.yaml"}\
#CKPT=${2:-"pretrained_models/fast_res50_256x192_icu.pth"}\
#OUTDIR=${3:-"data/res3"}\
#python scripts/demo_api_copy.py --cfg "configs/coco/custom/256x192_res50_lr1e-3_1x.yaml" --checkpoint "pretrained_models/fast_res50_256x192_icu.pth" --outdir "data/res3" --detector yolox-l-icu --detbatch 3 --posebatch 3 --eval --vis_fast --vis --sp #--action  # --save_video --vis_fast --save_img --showbox\