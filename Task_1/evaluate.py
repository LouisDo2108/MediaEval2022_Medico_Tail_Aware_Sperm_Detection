import numpy as np
from natsort import natsorted
from pathlib import Path
import os
import torch
import argparse
from ByteTrack.exps.example.mot.visem import Exp
from ByteTrack.yolox.evaluators.mot_evaluator import MOTEvaluator
from ByteTrack_utils.utils.get_tracking_metrics import get_tracking_metrics

# import json
# import cv2
# import glob
# from dataclasses import dataclass
# import importlib.util
# import sys


# @dataclass
# class Args:
#     def __init__(self, track_thresh=0.6, track_buffer=30, mot20=False, match_thresh=0.8):
#         self.track_thresh = track_thresh
#         self.track_buffer = track_buffer
#         self.mot20 = mot20
#         self.match_thresh = match_thresh
#         self.min_box_area = 10


def make_parser():
    parser = argparse.ArgumentParser("Evaluate VISEM data using ByteTrack")
    parser.add_argument("--data_dir", type=str, default="../VISEM_Tracking_Train_v4/",
        help="VISEM data root directory", nargs='?')
    parser.add_argument("--result_dir", type=str, default="../result/",
        help="A directory to store resulted files (The same as prepare_data.py's)", nargs='?')
    parser.add_argument("--gt_dir", type=str, default='../gt/',
                        help="A directory for storing validation ground truth files (The same as prepare_data.py's)", nargs='?')
    parser.add_argument("--yolo_model_path", type=str,
                        default=None, help="Path to the trained yolo model")
    parser.add_argument("--track_thresh", type=float, default=0.6,
                        help="tracking confidence threshold", nargs='?')
    parser.add_argument("--track_buffer", type=int, default=30,
                        help="the frames for keep lost tracks", nargs='?')
    parser.add_argument("--match_thresh", type=float, default=0.9,
                        help="matching threshold for tracking", nargs='?')
    parser.add_argument("--min-box-area", type=float,
                        default=100, help='filter out tiny boxes', nargs='?')
    parser.add_argument("--mot20", dest="mot20", default=False,
                        action="store_true", help="test mot20.")
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    args.data_dir = str(Path(args.data_dir).resolve())
    args.result_dir = str(Path(args.result_dir).resolve())
    args.gt_dir = str(Path(args.gt_dir).resolve())

    # Load yolov5 trained model
    MODEL_PATH = args.yolo_model_path
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=MODEL_PATH)  # , device='cpu'

    # Create dataloader
    train_ann = os.path.join(args.data_dir, 'annotations/Train.json')
    val_ann = os.path.join(args.data_dir, 'annotations/Val.json')
    exp = Exp(args.data_dir, train_ann, val_ann, img_size=640)
    dataloader = exp.get_eval_loader(batch_size=1, is_distributed=False)

    # Create an MOTEvaluator object and run evaluate
    evaluator = MOTEvaluator(dataloader=dataloader, img_size=[
                             640, 640], confthre=0.1, nmsthre=0.7, num_classes=3, args=args)

    # Run evaluation
    Path(args.gt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents = True, exist_ok = True)
    result=evaluator.evaluate(model, result_folder = args.result_dir)
    print("Detection result")
    print(result)

    with open(os.path.join(args.result_dir, "detection.txt"), "w") as text_file:
        text_file.write(result)
    print("Saved result to {}".format(
        os.path.join(args.result_dir, "detection.txt")))

    summary=get_tracking_metrics(args.result_dir, args.gt_dir)
    print(summary)

    with open(os.path.join(args.result_dir, "tracking.txt"), "w") as text_file:
        text_file.write(summary)
    print("Saved result to {}".format(
        os.path.join(args.result_dir, "tracking.txt")))
