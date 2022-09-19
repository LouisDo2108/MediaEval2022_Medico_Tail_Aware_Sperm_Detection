import fnmatch
from lib2to3.pgen2.token import RPAR
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm


def scan_dir(root, pattern):
    """Scan directory and find file that match pattern

    Args:
        root (str): path of directory to begin scanning
        pattern (str): pattern to filter for

    Yields:
        str: Full path to the file
    """
    for dirpath, _, files in os.walk(root):
        files = fnmatch.filter(files, pattern)
        if len(files) == 0:
            continue
        for filename in files:
            yield os.path.join(dirpath, filename)


def num_of_frame(dir_path, video_id):
    _dir = os.path.join(dir_path, f"{video_id}/labels_ftid/")
    print(_dir)
    list_frame = list(scan_dir(_dir, "*.txt"))
    return len(list_frame)


def frame_path(dir, video_id, frame_id):
    path = os.path.join(dir, f"{video_id}/labels_ftid/",
                        f"{video_id}_frame_{frame_id}_with_ftid.txt")
    if os.path.exists(path):
        return path
    return None


def extract_from_frame(path):
    df = pd.read_csv(path,
                     sep=" ",
                     header=None,
                     names=["name", "label", "x", "y", "w", "h"])
    data = {}
    for row in df.iterrows():
        data[row[1]['name']] = dict((key, val) for key, val in row[1].items())
    return data


def make_sperm_info(sperm_history):
    n_frame = len(sperm_history)
    total_dis = 0.
    last_x = None
    last_y = None
    first_x, first_y = None, None
    for frame_id, frame_info in sperm_history:
        center_x, center_y = frame_info['x'], frame_info['y']

        if any([first_x is None, first_y is None]):
            first_x = center_x
            first_y = center_y

        if all([last_x is not None, last_y is not None]):
            total_dis += np.sqrt((center_x - last_x)**2 +
                                 (center_y - last_y)**2)
        last_x = center_x
        last_y = center_y
        pass
    vector_distance = np.sqrt((first_x - last_x)**2 + (first_y - last_y)**2)
    return total_dis, vector_distance, total_dis / n_frame


def extract_feature(dir_path, video_id, num_frame):
    data = {}
    for frame_id in tqdm(range(num_frame), desc=f"Reading ID-{video_id}..."):
        path = frame_path(dir_path, video_id=video_id, frame_id=frame_id)
        if path is None: continue
        frame_data = extract_from_frame(path)
        for name, info in frame_data.items():
            if name not in data.keys(): data[name] = []
            data[name].append([frame_id, info])

    # Take sperm-id
    names = list(data.keys())
    avg_speed = 0.
    avg_dis = 0.
    avg_vector_distance = 0.
    n_name = len(names)
    # Aggregate speed and distance
    for name in tqdm(names):
        _total_dis, _total_vector_distance, _avg_speed = make_sperm_info(
            data[name])
        avg_dis += _total_dis
        avg_vector_distance += _total_vector_distance
        avg_speed += _avg_speed
        pass
    avg_dis /= n_name
    avg_vector_distance /= n_name
    avg_speed /= n_name

    return n_name, avg_dis, avg_vector_distance, avg_speed


if __name__ == "__main__":
    ID = 11
    DIR = f"./data/archive/VISEM_Tracking_Train_v4/Train"
    FRAME = 0
    PATH = f"{DIR}/{ID}_frame_{FRAME}_with_ftid.txt"

    num_frame = num_of_frame(f"./data/archive/VISEM_Tracking_Train_v4/Train/",
                             video_id=11)
    print(num_frame)
    print(extract_feature(dir_path=DIR, video_id=11, num_frame=num_frame))
    pass