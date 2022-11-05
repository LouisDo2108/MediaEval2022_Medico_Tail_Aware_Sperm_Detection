import argparse
import cv2
from pathlib import Path
import os
from visem_utils import get_video_splits
from natsort import natsorted
import torch
import shutil

def write_video():
    pass

def create_folder_template(sub_video_path):
    sub_video_path.mkdir(exist_ok=True)
    sub_video_images_path = sub_video_path / 'images'
    sub_video_images_path.mkdir(exist_ok=True)
    sub_video_labels_path = sub_video_path / 'labels'
    sub_video_labels_path.mkdir(exist_ok=True)
    sub_video_labels_ftid_path = sub_video_path / 'labels_ftid'
    sub_video_labels_ftid_path.mkdir(exist_ok=True)
    return sub_video_images_path

def split_video(video_path, model, output_path):
    
    id = Path(video_path).stem
    Path(output_path).mkdir(exist_ok=True)
    save_dir = Path(output_path) / id
    save_dir_images_path = create_folder_template(save_dir)
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
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
    vid_path, vid_writer = None, None
    
    if len(splits) > 0:
        
        print("Splits found")
        last_frame = 0
        
        for idx, split in enumerate(splits):
            
            vid_path = str(Path(output_path) / "{}_{}.mp4".format(id, idx) )
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()  # release previous video writer
            vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            for j, k in enumerate(range(last_frame, split)):
                img = cv2.imread(list_of_frames[k])
                vid_writer.write(img)
            
            last_frame = split
            
            if idx == len(splits) - 1:
                vid_path = str(Path(output_path) / "{}_{}.mp4".format(id, idx+1))
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                for frame in list_of_frames[last_frame:]:
                    img = cv2.imread(frame)
                    vid_writer.write(img)
                
    print("Removing folder")
    shutil.rmtree(save_dir, ignore_errors=True)
    print(10*'-', 'Done', 10*'-')
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='yolo model .pt path')
    parser.add_argument('--source', type=str, help='source')
    parser.add_argument('--output-path', type=str, help='output_path')
    opt = parser.parse_args()
    
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', opt.weights)
    split_video(
        video_path=opt.source,
        model=model,
        output_path=opt.output_path
    )
    