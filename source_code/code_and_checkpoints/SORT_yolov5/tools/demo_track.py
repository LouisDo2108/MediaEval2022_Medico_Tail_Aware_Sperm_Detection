
import argparse

import numpy as np
import cv2

from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

import sys
import os
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import logging
from SORT_tracker.sort import SORT

# from yolov5.models.common import DetectMultiBackend


from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from tools.visualization import plot_tracking

from pathlib import Path


# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 sort root directory
WEIGHTS = ROOT / 'weights'


# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

import numpy as np
from scipy.signal import find_peaks
from skimage.feature import hog
import copy

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def get_lines(img):
    low_threshold = 255/3
    high_threshold = 255
    img = apply_brightness_contrast(img, brightness = 0, contrast = 32)
    edge = cv2.Canny(img, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 3  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edge, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    if lines is not None:
        return len(lines)
    else:
        return 0


def get_tiles_center(cx, cy, rx, ry):
    tiles_center = [0 for x in range(9)]
    tiles_center[0] = (cx-rx, cy-ry)
    tiles_center[1] = (cx,   cy-ry)
    tiles_center[2] = (cx+rx, cy-ry)
    tiles_center[3] = (cx-rx, cy)
    tiles_center[4] = (cx,   cy)
    tiles_center[5] = (cx+rx, cy)
    tiles_center[6] = (cx-rx, cy+ry)
    tiles_center[7] = (cx,   cy+ry)
    tiles_center[8] = (cx+rx, cy+ry)
    return tiles_center

def get_roi_tiles(img, bbox, r=15, visualize=False, visualize_tiles=False):

    # image = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    x1, y1, x2, y2 = bbox#[int(x) for x in ccwh2xyxy_single(480, 640, bbox)]
    # cx, cy, w, h = get_cxcywh(480, 640, bbox)
    cx, cy, w, h = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
    rx = w/2+r
    ry = h/2+r
    tiles_center = get_tiles_center(cx, cy, rx, ry)
    xlim = 640*0.025
    ylim = 480*0.025

    xmin, xmax, ymin, ymax = round(cx-w/2-r), round(cx+w/2+r), round(cy-h/2-r), round(cy+h/2+r)
    if xmin <= 0: 
        xmin = 0
    if ymin <= 0:
        ymin = 0
    if ymax >= 480:
        ymax = 480
    if xmax >= 640:
        xmax = 640

    if visualize:
        img_ = copy.deepcopy(img)
        img_ = cv2.rectangle(img_, (x1, y1), (x2, y2), (0, 0, 0), thickness=1)
        img_ = cv2.rectangle(img_, (xmin, ymin), (xmax, ymax), (0, 0, 0), thickness=1)
        plt.subplot(111)
        plt.imshow(img_, cmap='gray')
        plt.gcf().set_size_inches(8, 12)
        plt.show()
        plt.subplot(111)
        plt.imshow(img[ymin:ymax, xmin:xmax], cmap='gray')
        plt.gcf().set_size_inches(8, 12)
        plt.show()

    tiles = dict.fromkeys([x for x in range(9)])

    for idx, (x, y) in enumerate(tiles_center):
        if x < 0 or x > 640 or y < 0 or y > 480:
            if (abs(x) < xlim or abs(x) < 640 + xlim) and (abs(y) < ylim or abs(y) < 480 + ylim):
                tiles[idx] = 0
                # print("Skip tile", idx)
                continue

        x1 = round(x - r)
        x2 = round(x + r)
        y1 = round(y - r)
        y2 = round(y + r)

        if x1 <= xmin: 
            x1 = xmin
        if y1 <= ymin: 
            y1 = ymin
        if x2 >= xmax:
            x2 = xmax
        if y2 >= ymax:
            y2 = ymax
        tiles[idx] = img[y1:y2, x1:x2]

    return tiles

def check_tail(hog_array):

    result = hog_array
    tail = False

    for ix, (k, res) in enumerate(result.items()):
        
        peaks, peak_plateaus = find_peaks(res, plateau_size=1)
        
        if len(peaks) < 1:
            continue
        if np.count_nonzero(np.diff(res)==0) >= 4:
            continue 
        candidate_peak = False
        for peak in peaks:
            if res[peak] >= 0.6:
                candidate_peak = True
                break
        if not candidate_peak:
            continue
        tail = True
        
    return tail


def get_image_gradient_histogram(img, visualize=False):
    
    low_threshold = 255/3
    high_threshold = 255
    img = cv2.GaussianBlur(img, (5, 5), 0)
    edge = cv2.Canny(img, low_threshold, high_threshold)
    fd, hog_image = hog(edge, orientations=9, pixels_per_cell=img.shape,
                    cells_per_block=(1, 1), visualize=True)
    

    if visualize:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(img, cmap='gray')
        ax1.set_title('Input image')

        # Rescale histogram for better display
        # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image, cmap='gray')
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show() 
    return fd


@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        img_size=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        eval=False,  # run multi-gpu eval
        is_download = False,
        suppress=False
):
    # speedlines = [[[0, 320], [1920, 320]],
    #     [[0, 720], [1920, 720]]]
    # countline = [[0, 520], [1920, 520]]
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file and is_download:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
    # save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    save_dir = increment_path(Path(project) / Path(source).stem, exist_ok=exist_ok)
    (save_dir / 'labels_ftid' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    if eval:
        device = torch.device(int(device))
    else:
        device = select_device(device)
    # model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    # print(yolo_weights)
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', str(yolo_weights[0]), force_reload=False)
    
    # stride, names, pt = model.stride, model.names, model.pt
    # img_size = check_img_size(img_size, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)
    
    if half:
        model.half()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=img_size, stride=stride)#, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # Create as many strong sort instances as there are video sources
    sort_tracker = SORT(speedlines=None, countline=None)
    outputs = [None] * nr_sources

    old_img_w = old_img_h = img_size
    old_img_b = 1
    # Run tracking
    # model.warmup(img_size=(1 if pt else nr_sources, 3, *img_size))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1
        
        # Warmup
        if device.type != 'cpu' and (old_img_b != im.shape[0] or old_img_h != im.shape[2] or old_img_w != im.shape[3]):
            old_img_b = im.shape[0]
            old_img_h = im.shape[2]
            old_img_w = im.shape[3]
            for i in range(3):
                model(im, augment=augment)[0]

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment)[0]#, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = '{}_frame_{}'.format(p.stem, frame_idx)
                    save_path = str(save_dir / "{}_tracking".format(p.stem))  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.stem / "{}_tracking".format(p.stem))  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0
            txt_path = str(save_dir / 'labels_ftid' / txt_file_name)  # im.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] 
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=1, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                
                # Suppress
                # Suppression
                if suppress:
                    # st = time.perf_counter()
                    det_ = det.cpu().numpy()[:, :4]
                    to_del = np.ones(det.shape[0])
                    for ix, (x1, y1, x2, y2) in enumerate(det_):
                        if det[ix, 4] >= 0.7 or det[ix, -1] != 0:
                            continue
                        cx, cy = (x1+x2)/2, (y1+y2)/2
                        if cx < im.shape[3]*0.025 or cx > im.shape[3]*0.975 or cy < im.shape[2]*0.025 or cy > im.shape[2]*0.975:
                            to_del[ix] = -1
                        elif cx < im.shape[3]*0.05 or cx > im.shape[3]*0.95 or cy < im.shape[2]*0.05 or cy > im.shape[2]*0.95:
                            to_del[ix] = 0 # Candidate

                    for d, v in enumerate(to_del):
                        if v == -1 or v == 1:
                            continue
                        bbox = det_[d, :4]
                        tiles = get_roi_tiles(im0, bbox=bbox, visualize=False)
                        result = {}
                        
                        lines = False
                        for idx, tile in tiles.items():
                            if idx == 4:
                                continue
                            if isinstance(tile, np.ndarray) and tile.shape[0] * tile.shape[1] >= 100:
                                if get_lines(tile) > 0:
                                    lines = True
                                    break
                                result[idx] = get_image_gradient_histogram(img=tile, visualize=False)
                        
                        if lines:
                            tail = True
                        else:
                            tail = check_tail(result)
                            
                        if tail:
                            to_del[d] = 1
                            
                    det = det[to_del == 1]
                    
                    # en = time.perf_counter()
                    # print("Extra time taken: {}".format(round(en-st, 2)))
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort 
                t4 = time_sync()
                outputs[i] = sort_tracker.update(frame_idx,det.cpu())
     
                t5 = time_sync()
                dt[3] += t5 - t4
                # print(outputs)
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                        bboxes = output.bbox
                        conf = output.conf
                        cls = output.class_id
                        id = output.track_id
                        # speed = output.speed
                        if save_txt:
                            xywh = (xyxy2xywh(torch.tensor(output.bbox.astype(np.float16)).view(1, 4)) / gn).view(-1).tolist()
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 6 + '\n') % (id, int(cls), *xywh))
                                # f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                #                                bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = (f'{id} {names[c]} {conf:.2f}') 
                            # #None if hide_labels else (
                            #     f'{id} {names[c]}' if hide_conf else \
                            #     (f'{id} {conf:.2f}' if hide_class else \
                            #     (f'{id} {names[c]} {speed:.2f} km/h' if speed > 0 else \
                            #     (f'{id} {names[c]} {conf:.2f}'))))
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                # sort_list[i].increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            # cv2.putText(im0, str(frame_idx), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (8, 150, 8),3)
            # cv2.putText(im0, "CAR COUNT : "+str(sort_tracker.cars), (60, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),3)
            # cv2.putText(im0, "MOTORBIKE COUNT : "+str(sort_tracker.motors), (60, 750), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),3)
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

    # Print results
    # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *img_size)}' % t)
    if save_txt or save_vid:
        # s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {str(save_dir / p.name)}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--img_size', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--is_download', default=False, action='store_true', help='download stream video')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--eval', action='store_true', help='run evaluation')
    parser.add_argument('--suppress', action='store_true', help='suppress instances near the border of the image')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)