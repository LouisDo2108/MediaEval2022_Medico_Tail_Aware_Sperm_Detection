#!/bin/bash
#SBATCH -o %j.out
#SBATCH --gres=gpu:1

conda init bash
conda source activate python38

CUDA_LAUNCH_BLOCKING=1 python3 /home/htluc/mediaeval2022_medico/yolov5/train.py \
	--img 640 --batch 32 --epochs 100 \
       --data /home/htluc/mediaeval2022_medico/yolov5/train_val.yaml \
       --weights yolov5m.pt
