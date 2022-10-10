import os
import numpy as np
import argparse
from pathlib import Path
from YoloV5_utils.utils import create_yaml
from sklearn.model_selection import train_test_split

def make_parser():
    parser = argparse.ArgumentParser(
        "Prepare data for Training YoloV5")
    parser.add_argument("--data_dir", type=str, default='../VISEM_Tracking_Train_v4/',
                        help='VISEM data root directory', nargs='?')
    parser.add_argument("--test_size", type=float, default=0.2,
                        help='The validation/test ratio', nargs='?')
    return parser

def create_val_dir(args):
    num = np.sort(
        np.array(os.listdir(os.path.join(args.data_dir, 'Train')), dtype='int32'))
    train_num, val_num = train_test_split(num, test_size=0.2, random_state=42)

    # Create a validation folder
    Path(os.path.join(args.data_dir, 'Val')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.data_dir, 'annotations')).mkdir(
        parents=True, exist_ok=True)
    dest = os.path.join(args.data_dir, 'Val')

    for folder in val_num:
        source = os.path.join(args.data_dir, 'Train', str(folder))
        os.system(f'mv {source} {dest}')


if __name__ == "__main__":
    
    args = make_parser().parse_args()
    args.data_dir = str(Path(args.data_dir).resolve())
    
    if not os.path.exists(os.path.join(args.data_dir, 'Val')) or \
        len(os.listdir(os.path.join(args.data_dir, 'Val'))) == 0:
        create_val_dir(args)

    create_yaml(args.data_dir)