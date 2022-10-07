import numpy as np
import os
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from ByteTrack_utils.utils.create_annotations import create_annotations
from ByteTrack_utils.utils.create_val_gt import create_val_gt


def make_parser():
    parser = argparse.ArgumentParser(
        "Prepare data for evaluate YoloV5 using ByteTrack")
    parser.add_argument("--root_dir", type=str, default='./VISEM_Tracking_Train_v4/',
                        help='The data root directory', nargs='?')
    parser.add_argument("--gt_dir", type=str, default='./gt/',
                        help='A directory for storing validation gt files', nargs='?')
    parser.add_argument("--test_size", type=float, default=0.2,
                        help='The validation/test ratio', nargs='?')
    return parser


def create_val_dir(args):
    num = np.sort(
        np.array(os.listdir(os.path.join(args.root_dir, 'Train')), dtype='int32'))
    train_num, val_num = train_test_split(num, test_size=0.2, random_state=42)

    # Create a validation folder
    Path(os.path.join(args.root_dir, 'Val')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.root_dir, 'annotations')).mkdir(
        parents=True, exist_ok=True)
    dest = os.path.join(args.root_dir, 'Val')

    for folder in val_num:
        source = os.path.join(args.root_dir, 'Train', str(folder))
        os.system(f'mv {source} {dest}')


def create_anno_json(args):
    ANNOTATIONS_PATH = os.path.join(args.root_dir, 'annotations')
    img_h, img_w = create_annotations(args.root_dir, ANNOTATIONS_PATH)
    return img_h, img_w


def create_gt(args, img_h, img_w):
    VAL_PATH = os.path.join(args.root_dir, 'Val')
    create_val_gt(VAL_PATH, args.gt_dir, img_h, img_w)


if __name__ == "__main__":

    args = make_parser().parse_args()

    # Create a validation folder
    create_val_dir(args)

    # Create annotation json files
    img_h, img_w = create_anno_json(args)

    # Create ground truth files for evaluation
    create_gt(args, img_h, img_w)
