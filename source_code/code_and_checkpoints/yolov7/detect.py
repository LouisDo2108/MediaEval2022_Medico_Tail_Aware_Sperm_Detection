import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

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


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    save_dir = Path(increment_path(Path(opt.project) / Path(opt.source).stem, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', weights[0], force_reload=False)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # if trace:
    #     model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_frame_{int(frame)-1}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Suppression
                if opt.suppress:
                    st = time.perf_counter()
                    det_ = det.cpu().numpy()[:, :4]
                    to_del = np.ones(det.shape[0])
                    for ix, (x1, y1, x2, y2) in enumerate(det_):
                        if det[ix, 4] >= 0.7 or det[ix, -1] != 0:
                            continue
                        cx, cy = (x1+x2)/2, (y1+y2)/2
                        if cx < img.shape[3]*0.025 or cx > img.shape[3]*0.975 or cy < img.shape[2]*0.025 or cy > img.shape[2]*0.975:
                            to_del[ix] = -1
                        elif cx < img.shape[3]*0.05 or cx > img.shape[3]*0.95 or cy < img.shape[2]*0.05 or cy > img.shape[2]*0.95:
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
                    
                    en = time.perf_counter()
                    print("Extra time taken: {}".format(round(en-st, 2)))
                

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                txt_result = []
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        line = [cls.item(), *xywh, conf] if opt.save_conf else [cls.item(), *xywh]
                        txt_result.append(line)
                        # with open(txt_path + '.txt', 'a') as f:
                        #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                txt_result = np.array(txt_result)
                np.savetxt(txt_path + '.txt', txt_result, delimiter=' ', fmt='%g')

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--suppress', action='store_true', help='suppress instances near the border of the image')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
