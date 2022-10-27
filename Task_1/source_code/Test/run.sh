#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate python38

CUDA_LAUNCH_BLOCKING=1 
python /home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/prepare_data_eval.py \
--data_dir /home/htluc/datasets/VISEM_Tracking_Train_v4/ \
--gt_dir /home/htluc/datasets/VISEM_Tracking_Train_v4/gt/ \
--task 'Test'

python /home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/video2img.py

python /home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/test_yolov7.py \
--data_dir /home/htluc/datasets/VISEM_Tracking_Train_v4/ \
--gt_dir /home/htluc/datasets/VISEM_Tracking_Train_v4/gt/ \
--result_dir /home/htluc/mediaeval2022_medico/yolov7/result \
--yolo_model_path /home/htluc/mediaeval2022_medico/yolov7/result/yolov7_img_weights_aug_custom_yaml/weights/best.pt \
--img_size 640