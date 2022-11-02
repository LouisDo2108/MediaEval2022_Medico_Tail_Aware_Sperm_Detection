#!/bin/bash
#SBATCH -o %j.out
#SBATCH --gres=gpu:1
#SBATCH --nodelist=selab2
#SBATCH --mem-per-cpu=4GB

eval "$(conda shell.bash hook)"
conda activate medico

export PYTHONPATH="${PYTHONPATH}:/home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/submission_code/code/ByteTrack"
source ~/.bashrc

eval "$(conda shell.bash hook)"
conda activate medico

python /home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/submission_code/code/video2img.py \
--source /home/htluc/datasets/VISEM_Tracking_Train_v4/Val/11/11.mp4 \
--weights /home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/submission_code/models/yolov7.pt

# Yolov7 detection
CUDA_LAUNCH_BLOCKING=1  python "/home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/submission_code/code/yolov7/detect.py" \
--conf 0.25 --img-size 640 \
--source /home/htluc/datasets/VISEM_Tracking_Train_v4/Val/11/11.mp4 \
--weights /home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/submission_code/models/yolov7.pt \
--project /home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/submission_code/predictions/ \
--exist-ok --no-trace --save-txt

# python /home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/submission_code/code/prepare_data_test.py \
# --data_path /home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/submission_code/data \
# --output_path /home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/submission_code/annotations

# ByteTrack tracking
# CUDA_LAUNCH_BLOCKING=1 python "/home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/submission_code/code/test_yolov7.py" \
# --result_dir "/home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/submission_code/result" \
# --test_ann "/home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/submission_code/annotations/Test.json" \
# --yolo_model_path '/home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/submission_code/models/best.pt' \
# --video_path "/home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/submission_code/videos/66.mp4" \
# --img_size 640