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
import cv2
from collections import defaultdict
from loguru import logger
from tqdm import tqdm

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# from ByteTrack.exps.example.mot.visem import Exp
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
                # xywh = tlwh.view(-1).tolist()
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))
    
                # for frame_id, tlwhs, track_ids, scores in results:
            # for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
            #     if track_id < 0:
            #         continue
            #     x1, y1, w, h = tlwh
            #     line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
            #     f.write(line)
                
            # for *xyxy, conf, cls in reversed(det):
            #         if save_txt:  # Write to file
            #             xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            #             line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
            #             with open(txt_path + '.txt', 'a') as f:
            #                 f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # if save_img or view_img:  # Add bbox to image
            #     label = f'{names[int(cls)]} {conf:.2f}'
            #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            
            # if vid_path != save_path:  # new video
            #     vid_path = save_path
            #     if isinstance(vid_writer, cv2.VideoWriter):
            #         vid_writer.release()  # release previous video writer
            #     if vid_cap:  # video
            #         fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #         w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #         h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #     else:  # stream
            #         fps, w, h = 30, im0.shape[1], im0.shape[0]
            #         save_path += '.mp4'
            #     vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            # vid_writer.write(im0)


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
    
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from ByteTrack.yolox.exp import Exp as MyExp
from ByteTrack.yolox.data import get_yolox_datadir

class Exp(MyExp):
    def __init__(self, data_dir, train_ann, val_ann, img_size, num_classes=3):
        super(Exp, self).__init__()
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.train_ann = train_ann,
        self.val_ann = val_ann
        self.input_size = (img_size, img_size)#(640, 640)
        self.test_size = (img_size, img_size)#(640, 640)
        self.random_size = (18, 32)
        self.max_epoch = 10
        self.print_interval = 20
        self.eval_interval = 5
        self.test_conf = 0.1
        self.nmsthre = 0.7
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from ByteTrack.yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset = MOTDataset(
            # data_dir=os.path.join(get_yolox_datadir(), "ch_all"),
            data_dir=self.data_dir,#"/content/drive/MyDrive/MediaEval2022_Medico/VISEM_Tracking_Train_v4/",
            json_file=self.train_ann,
            name='',
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from ByteTrack.yolox.data import MOTDataset, ValTransform

        valdataset = MOTDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            img_size=self.test_size,
            name='train',
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from ByteTrack.yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator



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
            
        # vid_writer = cv2.VideoWriter(
        # save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))
            
        tracker = BYTETracker(self.args)
        ori_thresh = self.args.track_thresh
         
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
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
                print(outputs.pred[0])
                # return
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
    # parser.add_argument("--data_dir", type=str, default="../VISEM_Tracking_Train_v4/",
    #     help="VISEM data root directory")
    parser.add_argument("--result_dir", type=str, default="../result/",
        help="A directory to store resulted files")
    parser.add_argument("--test_ann", type=str, help="Test annotation JSON")
    parser.add_argument("--video_path", type=str, help="Video path")
    parser.add_argument("--fps", type=str, help="Video path", nargs='?')
    # parser.add_argument("--gt_dir", type=str, default='../gt/',
    #                     help="A directory for storing validation ground truth files (The same as prepare_data_eval.py's)", nargs='?')
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
    args.result_dir = str(Path(args.result_dir).resolve())
    args.video_path = str(Path(args.video_path).resolve())

    # Load yolov5 trained model
    MODEL_PATH = args.yolo_model_path
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', MODEL_PATH)
    
    video = cv2.VideoCapture(args.video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    args.frame_rate = fps

    # Create dataloader
    # test_ann = args.test_ann#'/home/htluc/mediaeval2022_medico/MediaEval2022_Medico/Task_1/source_code/submission_code/annotations/Test.json'
    exp = Exp(None, None, args.test_ann, img_size=args.img_size)
    dataloader = exp.get_eval_loader(batch_size=1, is_distributed=False)

    # Create an MOTEvaluator object and run evaluate
    evaluator = MOTEvaluator(dataloader=dataloader, img_size=[
                             args.img_size, args.img_size], confthre=0.1, nmsthre=0.7, num_classes=3, args=args)

    # Run evaluation
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    evaluator.evaluate(model, result_folder=args.result_dir)