import pandas as pd
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
import torch
from volov5.utils.plots import Annotator, colors
from natsort import natsorted


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


def get_video_splits(id, model_path, data_path='/content/drive/MyDrive/MediaEval2022_Medico/VISEM_Tracking_Train_v4/', split='Train',
                     img_size=(480, 640), fps=49, min_frames_len=5, std_scale=3.5, crop=0.2,
                     candidate_ratio_lb=0.25, significant_ratio_lb=0.4,
                     visualize=False):

    h, w = img_size
    h_crop, w_crop = int(h*crop), int(w*crop)

    values = []
    data_path = Path(data_path) / id / 'images' / split
    imgs = [str(x) for x in data_path.iterdir()]
    imgs = natsorted(imgs)
    psnr_diff = 0
    lim = 99999
    last_frame = 0
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    result = []

    for idx, _ in enumerate(tqdm(imgs)):

        if idx + 1 == len(imgs) - 1:
            break

        if idx == 0:
            I = cv2.imread(imgs[idx])[h_crop:h-h_crop,
                                      w_crop:w-w_crop, :].flatten() / 255
            K = cv2.imread(imgs[idx+1])[h_crop:h-h_crop,
                                        w_crop:w-w_crop, :].flatten() / 255
        else:
            I = K
            K = cv2.imread(imgs[idx+1])[h_crop:h-h_crop,
                                        w_crop:w-w_crop, :].flatten() / 255
        mse = np.mean(np.power((I - K), 2))
        PSNR_Value = 10 * np.log10(1 / mse)

        if len(values[last_frame:]) > 0:
            psnr_diff = np.abs(PSNR_Value-np.mean(values[last_frame:]))
            lim = std_scale*np.std(values[last_frame:])

        if (idx-last_frame) > fps*min_frames_len and psnr_diff > lim and np.abs(1-PSNR_Value/values[-1]) >= candidate_ratio_lb:

            if (len(imgs) - idx) < fps*min_frames_len:
                values.append(PSNR_Value)
                continue

            result_1 = model(imgs[idx])
            result_2 = model(imgs[idx+1])

            if np.abs(1-PSNR_Value/values[-1]) <= significant_ratio_lb and np.abs(result_1.pred[0].shape[0] - result_2.pred[0].shape[0]) <= 1:
                values.append(PSNR_Value)
                continue

            if visualize:
                plt.subplot(121)
                plt.imshow(cv2.imread(imgs[idx]))
                plt.subplot(122)
                plt.imshow(cv2.imread(imgs[idx+1]))
                plt.show()

            print("Cut at frame", idx+1, " time: ", np.round(idx/fps, 2))
            print("PSNR-Obj: Frame {}: {}-{}; Frame {}: {}-{}".
                  format(idx, values[-1], result_1.pred[0].shape[0],  idx+1, PSNR_Value, result_2.pred[0].shape[0]))
            result.append(idx+1)
            last_frame = idx+1
            
        values.append(PSNR_Value)

    # temp = np.array(list(zip(np.arange(0, len(values)), values)))
    return result
