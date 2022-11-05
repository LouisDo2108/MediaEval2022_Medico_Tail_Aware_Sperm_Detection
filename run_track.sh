#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate medico

# SORT tracking
CUDA_LAUNCH_BLOCKING=1  
python ./code/track.py \
--img-size 640 \
--source ./videos/66.mp4 \
--yolo-weights ./models/yolov7.pt \
--project ./predictions/ \
--exist-ok --save-txt --save-vid 
# --suppress \