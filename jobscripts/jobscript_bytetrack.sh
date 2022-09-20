#!/bin/bash
#SBATCH -o %j.out
#SBATCH --gres=gpu:1
#conda init bash
#conda activate python38_bytetrack
CUDA_LAUNCH_BLOCKING=1 python3 '/home/htluc/mediaeval2022_medico/ByteTrack/tools/train.py' \
	-f '/home/htluc/mediaeval2022_medico/ByteTrack/exps/example/mot/visem.py' \
	-d 1 -b 4 --fp16 -o \
	-c '/home/htluc/mediaeval2022_medico/ByteTrack/pretrained/yolox_x.pth'
