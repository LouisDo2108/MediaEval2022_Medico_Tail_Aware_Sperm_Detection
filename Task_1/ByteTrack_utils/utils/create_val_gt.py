from ByteTrack_utils.utils.convert_bbox_format import ccwh2xyxy, xyxy2xywh
from ByteTrack_utils.utils.hashing_trackid import str_to_int
from natsort import natsorted
import glob
import os
import pandas as pd

def create_val_gt(val_path, output_path, height, width):
    for folder in natsorted(glob.glob(os.path.join(val_path, '*'))):
        print(folder)
        df = pd.DataFrame()
        frame_id = 1
        
        for f in natsorted(glob.glob(os.path.join(folder, 'labels_ftid', '*.txt'))):
            print(f)
            temp = pd.read_csv(
                f, names=['track_id', 'class', 'x', 'y', 'w', 'h'], sep=' ')
            bbox = temp.iloc[:, 2:].to_numpy()
            bbox = ccwh2xyxy(height, width, bbox)
            bbox = xyxy2xywh(bbox)
            temp.iloc[:, 2:] = bbox
            temp['frame_id'] = frame_id
            temp['track_id'] = temp['track_id'].map(str_to_int)
            df = pd.concat([df, temp], axis=0)
            frame_id = frame_id + 1

        df['conf'] = 1
        df['a'] = -1
        df['b'] = -1
        df['c'] = -1
        df = df[['frame_id', 'track_id', 'x',
                 'y', 'w', 'h', 'conf', 'a', 'b', 'c']]
        df = df.sort_values(['track_id', 'frame_id'], ascending=True)
        df.to_csv(os.path.join(output_path, '{}.txt'.format(
            folder.split('/')[-1])), index=False, sep=',', header=False)
