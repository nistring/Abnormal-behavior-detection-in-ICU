#!bin/bash
set -x

python ./tools/train.py -b 4 -f ./exps/example/custom/yolox_l_icu.py \
    -c ./data/yolox_l.pth \
    --fp16 -o