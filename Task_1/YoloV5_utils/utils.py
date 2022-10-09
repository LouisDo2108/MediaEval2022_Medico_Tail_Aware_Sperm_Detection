import numpy as np
import os
import glob
from pathlib import Path
import cv2
from tqdm.notebook import tqdm
# import matplotlib.pyplot as plt
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


def get_gt_video(id):
    """
    Given the id of the video, output the video with ground truth bbox.
    """
    id = id

    images_path = Path('/content/drive/MyDrive/MediaEval2022_Medico/VISEM_Tracking_Train_v4/Val/{}/images'.format(id))
    labels_path = Path('/content/drive/MyDrive/MediaEval2022_Medico/VISEM_Tracking_Train_v4/Val/{}/labels'.format(id))
    classes = ['sperm', 'cluster', 'small']
    images = natsorted([x for x in images_path.iterdir()])

    video_name = '/content/drive/MyDrive/MediaEval2022_Medico/{}gt.mp4'.format(id)

    frame = cv2.imread(str(images[0]))
    height, width, layers = frame.shape
    print(frame.shape)

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width,height))

    for image in tqdm(images):
        img = cv2.imread(str(image))
        annotator = Annotator(img, line_width=2, example=str('yolov5m'))
        name = image.stem
        labels = labels_path / image.stem
        labels = str(labels) + '.txt'
        # print(labels)
        if os.path.exists(labels) == True:
            anns = np.genfromtxt(labels, dtype='float32')
            anns = np.atleast_2d(anns)
            anns[:, 1:] = ccwh2xyxy(480, 640, anns[:, 1:])
            for ann in anns:
                cls = classes[int(ann[0])]
                annotator.box_label(ann[1:].round(), cls, color=colors(int(ann[0]), True))
                # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                # cv2.putText(img,cls,(x,y),0,0.3,(0,255,0))
        # plt.gcf().set_size_inches(6.4, 4.8)
        # plt.imshow(annotator.result())
        video.write(annotator.result())

    cv2.destroyAllWindows()
    video.release()