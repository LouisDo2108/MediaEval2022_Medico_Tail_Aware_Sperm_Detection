import argparse
import cv2
from pathlib import Path
import os
from visem_utils import get_video_splits
from natsort import natsorted
import torch
import shutil

def vid2img(video_path, model, output_path):
    
    id = Path(video_path).stem
    Path(output_path).mkdir(exist_ok=True)
    save_dir = Path(output_path) / id
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
            sub_video_path = Path(output_path) / "{}_{}".format(id, idx)
            sub_video_images_path = create_folder_template(sub_video_path)
                
            for j, k in enumerate(range(last_frame, split)):
                # dest = sub_video_path / "{}_frame_{}.jpg".format(id, j)
                shutil.move(list_of_frames[k], str(sub_video_images_path))
                print("Moving {} to {}".format(list_of_frames[k], str(sub_video_images_path))) 
            
            last_frame = split
        shutil.move(save_dir, str(Path(output_path) / "{}_{}".format(id, len(splits))))
        
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='yolo model .pt path')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    
    opt = parser.parse_args()
    
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', opt.weights)
    vid2img(
        video_path=opt.source,
        #"/home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/submission_code/videos/66.mp4", 
        model=model,
        output_path='/home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/submission_code/data/'
    )
    