import cv2
from pathlib import Path
import os
from YoloV5_utils.utils import get_video_splits
from natsort import natsorted
import torch
import shutil

def vid2img(video_path, model):
    
    id = Path(video_path).stem
    save_dir = Path(video_path).parent / id
    save_dir_images_path = create_folder_template(save_dir)
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    
    success, image = video.read()
    count = 0
    list_of_frames = []
    while success:
        cv2.imwrite(str(save_dir_images_path / "{}_frame_{}.jpg".format(id, count)), image)     # save frame as JPEG file      
        success,image = video.read()
        print('Reading frame:', count)
        list_of_frames.append(str(save_dir_images_path / "{}_frame_{}.jpg".format(id, count)))
        count += 1
        
    splits = get_video_splits(id=id, video_images_path=str(save_dir_images_path), model=model, fps=fps)
        
    if len(splits) > 0:
        
        print("Splits found")
        last_frame = 0
        
        for idx, split in enumerate(splits):
            sub_video_path = Path(video_path).parent / "{}_{}".format(id, idx)
            sub_video_images_path = create_folder_template(sub_video_path)
                
            for j, k in enumerate(range(last_frame, split)):
                # dest = sub_video_path / "{}_frame_{}.jpg".format(id, j)
                shutil.move(list_of_frames[k], str(sub_video_images_path))
                print("Moving {} to {}".format(list_of_frames[k], str(sub_video_images_path))) 
            
            last_frame = split
        shutil.move(save_dir, str(Path(video_path).parent / "{}_{}".format(id, len(splits))))
        
    print(10*'-', 'Done', 10*'-')

def create_folder_template(sub_video_path):
    sub_video_path.mkdir(exist_ok=True)
    sub_video_images_path = sub_video_path / 'images'
    sub_video_images_path.mkdir(exist_ok=True)
    sub_video_labels_path = sub_video_path / 'labels'
    sub_video_labels_path.mkdir(exist_ok=True)
    sub_video_labels_ftid_path = sub_video_path / 'labels_ftid'
    sub_video_labels_ftid_path.mkdir(exist_ok=True)
    return sub_video_images_path
    
if __name__ == "__main__":
    MODEL_PATH = '/home/htluc/mediaeval2022_medico/yolov7/result/yolov7_img_weights_aug_custom_yaml/weights/best.pt'
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', MODEL_PATH)
    # device = 'cuda' if torch.cuda.is_available else 'cpu'
    # model = torch.load(MODEL_PATH, map_location=device)
    # vid2img("/home/htluc/datasets/VISEM_Tracking_Train_v4/Test/66.mp4", model)
    vid2img("/home/htluc/datasets/VISEM_Tracking_Train_v4/Test/68.mp4", model)
    vid2img("/home/htluc/datasets/VISEM_Tracking_Train_v4/Test/73.mp4", model)
    vid2img("/home/htluc/datasets/VISEM_Tracking_Train_v4/Test/76.mp4", model)
    vid2img("/home/htluc/datasets/VISEM_Tracking_Train_v4/Test/80.mp4", model)
    