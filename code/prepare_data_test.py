import numpy as np
import os
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from visem_utils import create_annotations, create_val_gt


def make_parser():
    parser = argparse.ArgumentParser("Prepare data for testing")
    parser.add_argument("--data_path", type=str, help="video's images folder path")
    parser.add_argument("--output_path", type=str, help='folder to store annotation json')
    return parser


def create_anno_json(args):
    img_h, img_w = create_annotations(args.data_path, args.output_path)
    return img_h, img_w


if __name__ == "__main__":
    args = make_parser().parse_args()
    create_anno_json(args)
        
