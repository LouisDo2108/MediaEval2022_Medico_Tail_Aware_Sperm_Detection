import numpy as np
from natsort import natsorted
from pathlib import Path
import os
import torch
import argparse
from copy import deepcopy
import contextlib
import io
import itertools
import json
import tempfile
import time
from collections import defaultdict
from loguru import logger
from tqdm import tqdm

from ByteTrack.exps.example.mot.visem import Exp
from ByteTrack.yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        # video_names = defaultdict()
        video_names = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = BYTETracker(self.args)
        ori_thresh = self.args.track_thresh
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                print(info_imgs)
                video_id = info_imgs[3][0].split('/')[-1]
                img_file_name = info_imgs[4][0]
                video_name = video_id
                
                if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
                    self.args.track_buffer = 14
                elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
                    self.args.track_buffer = 25
                else:
                    self.args.track_buffer = 30

                if video_name == 'MOT17-01-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-06-FRCNN':
                    self.args.track_thresh = 0.65
                elif video_name == 'MOT17-12-FRCNN':
                    self.args.track_thresh = 0.7
                elif video_name == 'MOT17-14-FRCNN':
                    self.args.track_thresh = 0.67
                elif video_name in ['MOT20-06', 'MOT20-08']:
                    self.args.track_thresh = 0.3
                else:
                    self.args.track_thresh = ori_thresh

                if video_name not in video_names:
                    # video_names[video_id] = video_name
                    video_names.append(video_name)
                if frame_id == 1:
                    tracker = BYTETracker(self.args)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[-2]))
                        write_results(result_filename, results)
                        results = []

                if(torch.cuda.is_available()):
                    imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(info_imgs[-1][0])
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs.pred, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            if outputs.pred[0] is not None:
                img_info = [info_imgs[0].item(), info_imgs[1].item()]
                online_targets = tracker.update(outputs.pred[0][:, :5].cpu().numpy(), img_info, self.img_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                # save results
                results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[-1]))
                write_results(result_filename, results)
                
                
    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()
            bboxes = deepcopy(output[:, 0:4]) 

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 5]
            scores = output[:, 4]# * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": int(cls[ind]),#label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list


def make_parser():
    parser = argparse.ArgumentParser("Evaluate VISEM data using ByteTrack")
    parser.add_argument("--data_dir", type=str, default="../VISEM_Tracking_Train_v4/",
        help="VISEM data root directory", nargs='?')
    parser.add_argument("--result_dir", type=str, default="../result/",
        help="A directory to store resulted files", nargs='?')
    parser.add_argument("--gt_dir", type=str, default='../gt/',
                        help="A directory for storing validation ground truth files (The same as prepare_data_eval.py's)", nargs='?')
    parser.add_argument("--yolo_model_path", type=str,
                        default=None, help="Path to the trained yolo model")
    parser.add_argument("--img_size", type=int, default=640,
                        help="yolov5 training image size", nargs='?')
    parser.add_argument("--track_thresh", type=float, default=0.6,
                        help="tracking confidence threshold", nargs='?')
    parser.add_argument("--track_buffer", type=int, default=30,
                        help="the frames for keep lost tracks", nargs='?')
    parser.add_argument("--match_thresh", type=float, default=0.9,
                        help="matching threshold for tracking", nargs='?')
    parser.add_argument("--min-box-area", type=float,
                        default=100, help='filter out tiny boxes', nargs='?')
    parser.add_argument("--mot20", dest="mot20", default=False,
                        action="store_true", help="test mot20.")
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    args.data_dir = str(Path(args.data_dir).resolve())
    args.result_dir = str(Path(args.result_dir).resolve())
    args.gt_dir = str(Path(args.gt_dir).resolve())

    # Load yolov5 trained model
    MODEL_PATH = args.yolo_model_path
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', MODEL_PATH)

    # Create dataloader
    test_ann = os.path.join(args.data_dir, 'annotations/Test.json')
    exp = Exp(args.data_dir, None, test_ann, img_size=args.img_size)
    dataloader = exp.get_eval_loader(batch_size=1, is_distributed=False)

    # Create an MOTEvaluator object and run evaluate
    evaluator = MOTEvaluator(dataloader=dataloader, img_size=[
                             args.img_size, args.img_size], confthre=0.1, nmsthre=0.7, num_classes=3, args=args)

    # Run evaluation
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    evaluator.evaluate(model, result_folder=args.result_dir)