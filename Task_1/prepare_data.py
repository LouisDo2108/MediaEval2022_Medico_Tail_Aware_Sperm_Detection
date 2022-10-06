import numpy as np
import os
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from utils.create_annotations import create_annotations
from utils.create_val_gt import create_val_gt

def make_parser():
    parser = argparse.ArgumentParser("Prepare data for evaluate YoloV5 using ByteTrack")
    parser.add_argument("--root_dir", type=str, default='./VISEM_Tracking_Train_v4/', help='The data root directory', nargs='?')
    parser.add_argument("--gt_dir", type=str, default='./gt/', help='A directory for storing validation gt files', nargs='?')
    parser.add_argument("--test_size", type=float, default=0.2, help='The validation/test ratio', nargs='?')
    return parser

if __name__ == "__main__":
    
    args = make_parser().parse_args()
    
    DATA_PATH = args.root_dir
    num = np.sort(np.array(os.listdir(os.path.join(DATA_PATH, 'Train')), dtype='int32'))
    train_num, val_num = train_test_split(num, test_size=0.2, random_state=42)

    ### Create a validation folder
    Path(os.path.join(DATA_PATH, 'Val')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(DATA_PATH, 'annotations')).mkdir(parents=True, exist_ok=True)
    dest = os.path.join(DATA_PATH, 'Val')
    
    for folder in val_num:
        source = os.path.join(DATA_PATH, str(folder))
        os.command(f'mv {source} {dest}')
    
    ### Create annotation json files
    ANNOTATIONS_PATH = os.path.join(DATA_PATH, 'annotations')
    img_h, img_w = create_annotations(DATA_PATH, ANNOTATIONS_PATH)
    
    ### Create ground truth files for evaluation
    VAL_PATH = os.path.join(DATA_PATH, 'Val')
    create_val_gt(VAL_PATH, args.gt_dir, img_h, img_w)
    
    