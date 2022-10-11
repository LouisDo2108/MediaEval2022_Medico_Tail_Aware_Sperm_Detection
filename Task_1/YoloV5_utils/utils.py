import pandas as pd
from utils.plots import Annotator, colors, save_one_box
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from pathlib import Path
import cv2
from tqdm.notebook import tqdm
from ByteTrack_utils.utils.hashing_trackid import str_to_int
from ByteTrack_utils.utils.convert_bbox_format import ccwh2xyxy
from natsort import natsorted
from volov5.utils.plots import Annotator, colors


def create_yaml(main_train_dir):

    SAVE_PATH = os.path.join(main_train_dir, "train_val.yaml")

    with open(SAVE_PATH, "w") as f:

        # Writing train paths to yaml
        f.write("train: [ \n")
        for t in sorted(glob.glob(os.path.join(main_train_dir, "Train", "*"))):
            f.write(t + ",\n")
        f.write("]\n\n")

        # writing validation paths to yaml
        f.write("val: [\n")
        for v in sorted(glob.glob(os.path.join(main_train_dir, "Val", "*"))):
            f.write(v + ",\n")
        f.write("]\n\n")

        # writing number of class parameter
        f.write("nc: 3\n\n")

        # Writing class names
        f.write('names: [ "sperm", "cluster", "small_or_pinhead"]')
    f.close()


def get_gt_video(data_path, gt_dest_path, img_size=(480, 640), fps=49, folder='Train'):

    ids = [x for x in os.listdir('{}/{}'.format(data_path, folder))]
    h, w = img_size

    for id in tqdm(ids):
        images_path = Path(
            '{}/{}/{}/images'.format(data_path, folder, id))
        labels_path = Path(
            '{}/{}/{}/labels_ftid'.format(data_path, folder, id))
        classes = ['sperm', 'cluster', 'small']
        images = natsorted([x for x in images_path.iterdir()], key=str)

        video_name = '{}/{}_gt_videos/{}gt.mp4'.format(
            folder, id)
        print(video_name)
        frame = cv2.imread(str(images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(
            *'mp4v'), fps, (width, height))

        for image in tqdm(images):
            img = cv2.imread(str(image))
            annotator = Annotator(img, line_width=2, example=str('yolov5m'))
            name = image.stem
            labels = labels_path / image.stem
            labels = str(labels) + '_with_ftid.txt'
            if os.path.exists(labels) == True:
                anns = np.genfromtxt(labels, dtype='str')
                anns = np.atleast_2d(anns)
                track_id = np.array(
                    list(map(str_to_int, list(anns[:, 0]))), dtype='uint32')
                anns[:, 0] = track_id
                anns[:, 2:] = ccwh2xyxy(480, 640, anns[:, 2:].astype(
                    'float32')).round().astype('int32')
                for ann in anns:
                    cls = "{}{}{}".format(str(ann[0])[:2], str(
                        ann[0])[-2:], classes[int(ann[1])])
                    annotator.box_label(
                        ann[2:], cls, color=colors(int(ann[1]), True))
            video.write(annotator.result())

        cv2.destroyAllWindows()
        video.release()
        print("Saved to", video_name)
