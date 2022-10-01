import os
import torch
from yolox.tracker.byte_tracker import BYTETracker
from PIL import Image
from argparse import ArgumentParser


IMAGE_DIR = "data/archive/VISEM_Tracking_Train_v4/Train"

def make_image_frame_path(root, video_id, last_id):
    for id in range(last_id + 1):
        yield os.path.join(root, f"{video_id}/images/{video_id}_frame_{id}.jpg")

if __name__ == "__main__":
    detector = torch.hub.load('yolov5', 'custom', path="weight/yolov5m_best_01_10.pt", source='local')
    detector.eval()
    args = ArgumentParser()
    args.add_argument("--track_thresh", default=0.5)
    args.add_argument("--mot20", default=False)
    args.add_argument("--track_buffer", default=1)
    args.add_argument("--match_thresh", default=0.5)
    args = args.parse_args()
    tracker = BYTETracker(args)

    with torch.no_grad():
        for path in make_image_frame_path(IMAGE_DIR, 11, 1469):
            image = Image.open(path)
            pred = detector([image], size=640)
            names = pred.names
            xyxy = pred.xyxy[0] # -> [n_object, 6->(x1, y1, x2, y2, score, class)]
            dets = xyxy[:, :5]
            online_targets = tracker.update(dets.clone(), (640, 480), (640, 480))
            break