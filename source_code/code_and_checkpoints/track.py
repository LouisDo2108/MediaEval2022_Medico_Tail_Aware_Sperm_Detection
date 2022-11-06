from natsort import natsorted
import cv2
from pathlib import Path
import shutil
import argparse
from SORT_yolov5.tools.demo_track import run
from split_video import split_video
import copy
import torch
import os

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--suppress', action='store_true', help='suppress instances near the border of the image')
    opt = parser.parse_args()
    opt_ = copy.deepcopy(opt)
    
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', opt.yolo_weights[0])
    split_video(
        video_path=opt.source,
        model=model,
        output_path=str(Path(opt.source).parent)
    )
    
    id = Path(opt.source).stem
    video_path_list = []
    
    videos = natsorted(Path(opt.source).parent.glob(r'{}_*.mp4'.format(id)), key=str)
    if len(videos) > 1:
        for video_split_path_ in videos:
            if '_' in Path(video_split_path_).stem:
                print("There are scence changes in the video")
                video_split_path = str(video_split_path_.resolve())
                opt_.source = video_split_path
                run(**vars(opt_))
                video_path_list.append(str(Path(opt.project) / video_split_path_.stem / Path(video_split_path_).stem) + '_tracking.mp4')
    else:
        print("There is no scence change in the video")
        video_split_path = str(Path(opt.source).parent / "{}.mp4".format(id))
        run(**vars(opt_))
    save_path = str(Path(opt.project) / id) + "_tracking.mp4"
    
    if len(video_path_list) > 1:
        print("Merging videos")
        # Create a new video
        vid_cap = cv2.VideoCapture(video_path_list[0])
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(fps, w, h)
        video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        # Write all the frames sequentially to the new video
        for v in video_path_list:
            curr_v = cv2.VideoCapture(v)
            while curr_v.isOpened():
                r, frame = curr_v.read()    # Get return value and curr frame of curr video
                if not r:
                    break
                video.write(frame)          # Write the frame

        video.release() 
        print("Saved to ", save_path)
    
        save_dir = Path(opt.project) / id
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / 'labels_ftid').mkdir(parents=True, exist_ok=True)
        
        print("Reorganize")
        count = 0
        try:
            for v in video_path_list:
                label_path = Path(v).parent / 'labels_ftid'
                print("Moving labels from ", label_path, 'to', str(save_dir / 'labels_ftid'))
                for label_txt in natsorted(label_path.glob(r'*.txt'), key=str):
                    shutil.move(str(label_txt.resolve()), str(save_dir / 'labels_ftid' / (id + '_frame_{}.txt'.format(count))))
                    count += 1
            shutil.move(save_path, str(save_dir / "{}_tracking.mp4".format(id)))
        except Exception as e:
            print(e)
    
    save_dir = r"{}".format(opt.project)
    p = Path(save_dir).glob("*") 
    
    to_remove_folders = [str(x) for x in p if x.is_dir() and '_' in str(x.stem)]
    for folder in to_remove_folders:
        os.system("rm -rf {}".format(to_remove))
        print("Removed", to_remove)
    print("-"*10, "Done","-"*10)
    