#!/bin/bash
#SBATCH -o %j.out
#SBATCH --gres=gpu:1
#SBATCH --nodelist=selab3
#SBATCH --mem-per-cpu=4GB

eval "$(conda shell.bash hook)"
conda activate medico

export PYTHONPATH="${PYTHONPATH}:/home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/submission_code/code/ByteTrack"
source ~/.bashrc

eval "$(conda shell.bash hook)"
conda activate medico

# Convert video to images
# CUDA_LAUNCH_BLOCKING=1 
# python /home/htluc/mediaeval2022_medico/submission_code/code/video2img.py \
# --source /home/htluc/datasets/VISEM_Tracking_Train_v4/Val/11/11.mp4 \
# --weights /home/htluc/mediaeval2022_medico/submission_code/models/yolov7.pt

# Yolov7 detection
CUDA_LAUNCH_BLOCKING=1 
python /home/htluc/mediaeval2022_medico/submission_code/code/yolov7/detect.py \
--conf 0.25 --img-size 640 \
--source /home/htluc/datasets/VISEM_Tracking_Train_v4/Val/11/11.mp4 \
--weights /home/htluc/mediaeval2022_medico/submission_code/models/yolov7.pt \
--project /home/htluc/mediaeval2022_medico/submission_code/predictions_suppress/ \
--exist-ok --no-trace --save-txt --suppress