import cv2
import numpy as np
import json
from pathlib import Path
from zlib import crc32
from natsort import natsorted
import glob
import os
import pandas as pd
from pathlib import Path
from collections import OrderedDict
import motmetrics as mm
from loguru import logger
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

mm.lap.default_solver = 'lap'

class Annotator:
    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        self.pil = pil or non_ascii
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_pil_font(font='Arial.Unicode.ttf' if non_ascii else font,
                                       size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle(
                    (box[0], box[1] - h if outside else box[1], box[0] + w + 1,
                     box[1] + 1 if outside else box[1] + h + 1),
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im,
                            label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            self.lw / 3,
                            txt_color,
                            thickness=tf,
                            lineType=cv2.LINE_AA)

    def masks(self, masks, colors, im_gpu=None, alpha=0.5):
        """Plot masks at once.
        Args:
            masks (tensor): predicted masks on cuda, shape: [n, h, w]
            colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
            im_gpu (tensor): img is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float): mask transparency: 0.0 fully transparent, 1.0 opaque
        """
        if self.pil:
            # convert to numpy first
            self.im = np.asarray(self.im).copy()
        if im_gpu is None:
            # Add multiple masks of shape(h,w,n) with colors list([r,g,b], [r,g,b], ...)
            if len(masks) == 0:
                return
            if isinstance(masks, torch.Tensor):
                masks = torch.as_tensor(masks, dtype=torch.uint8)
                masks = masks.permute(1, 2, 0).contiguous()
                masks = masks.cpu().numpy()
            # masks = np.ascontiguousarray(masks.transpose(1, 2, 0))
            masks = scale_image(masks.shape[:2], masks, self.im.shape)
            masks = np.asarray(masks, dtype=np.float32)
            colors = np.asarray(colors, dtype=np.float32)  # shape(n,3)
            s = masks.sum(2, keepdims=True).clip(0, 1)  # add all masks together
            masks = (masks @ colors).clip(0, 255)  # (h,w,n) @ (n,3) = (h,w,3)
            self.im[:] = masks * alpha + self.im * (1 - s * alpha)
        else:
            if len(masks) == 0:
                self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
            colors = torch.tensor(colors, device=im_gpu.device, dtype=torch.float32) / 255.0
            colors = colors[:, None, None]  # shape(n,1,1,3)
            masks = masks.unsqueeze(3)  # shape(n,h,w,1)
            masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

            inv_alph_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
            mcs = (masks_color * inv_alph_masks).sum(0) * 2  # mask color summand shape(n,h,w,3)

            im_gpu = im_gpu.flip(dims=[0])  # flip channel
            im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # shape(h,w,3)
            im_gpu = im_gpu * inv_alph_masks[-1] + mcs
            im_mask = (im_gpu * 255).byte().cpu().numpy()
            self.im[:] = scale_image(im_gpu.shape, im_mask, self.im.shape)
        if self.pil:
            # convert im back to PIL and update draw
            self.fromarray(self.im)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor='top'):
        # Add text to image (PIL-only)
        if anchor == 'bottom':  # start y from font bottom
            w, h = self.font.getsize(text)  # text width, height
            xy[1] += 1 - h
        self.draw.text(xy, text, fill=txt_color, font=self.font)

    def fromarray(self, im):
        # Update self.im from a numpy array
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
colors = Colors()


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


def get_gt_video(video_images_path, gt_dest_path, img_size=(480, 640), fps=49, folder='Train'):

    ids = [x for x in os.listdir('{}/{}'.format(video_images_path, folder))]
    h, w = img_size

    for id in tqdm(ids):
        images_path = Path(
            '{}/{}/{}/images'.format(video_images_path, folder, id))
        labels_path = Path(
            '{}/{}/{}/labels_ftid'.format(video_images_path, folder, id))
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


def get_video_splits(id, model, video_images_path,
                     img_size=(480, 640), fps=49, min_frames_len=5, std_scale=3.5, crop=0.2,
                     candidate_ratio_lb=0.25, significant_ratio_lb=0.4,
                     visualize=False):

    h, w = img_size
    h_crop, w_crop = int(h*crop), int(w*crop)

    values = []
    video_images_path = Path(video_images_path)
    imgs = [str(x) for x in video_images_path.iterdir()]
    imgs = natsorted(imgs)
    psnr_diff = 0
    lim = 99999
    last_frame = 0
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
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
    return result


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names

def get_tracking_metrics(result_dir, gt_dir):
    gtfiles = glob.glob(os.path.join(gt_dir, '*.txt'))
    print('gt_files', gtfiles)
    tsfiles = [f for f in glob.glob(os.path.join(result_dir, '*.txt')) if not os.path.basename(f).startswith('eval')]

    logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logger.info('Loading files.')

    gt = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1)) for f in tsfiles])    

    mh = mm.metrics.create()    
    accs, names = compare_dataframes(gt, ts)

    logger.info('Running metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
                'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
                'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)

    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                        'partially_tracked', 'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logger.info('Completed')
    return summary

def bytes_to_int(b):
    return int(crc32(b) & 0xffffffff) #/ 2**32

def str_to_int(s, encoding="utf-8"):
    return bytes_to_int(s.encode(encoding))

def ccwh2xyxy_single(img_h, img_w, bb_box):
    norm_center_x = bb_box[0]
    norm_center_y = bb_box[1]
    norm_label_width = bb_box[2]
    norm_label_height = bb_box[3]

    center_x = norm_center_x * img_w
    center_y = norm_center_y * img_h
    label_width = norm_label_width * img_w
    label_height = norm_label_height * img_h

    x_min = center_x - (label_width/2)
    y_min = center_y - (label_height/2)
    x_max = center_x + (label_width/2)
    y_max = center_y + (label_height/2)

    return [x_min, y_min, x_max, y_max]


def xyxy2xywh_single(bboxes):
    bboxes[2] = bboxes[2] - bboxes[0]
    bboxes[3] = bboxes[3] - bboxes[1]
    return bboxes

def ccwh2xyxy(img_h, img_w, bb_box):
    norm_center_x = bb_box[:, 0]
    norm_center_y = bb_box[:, 1]
    norm_label_width = bb_box[:, 2]
    norm_label_height = bb_box[:, 3]
    
    center_x = norm_center_x * img_w
    center_y = norm_center_y * img_h
    label_width = norm_label_width * img_w
    label_height = norm_label_height * img_h
    
    x_min = center_x - (label_width/2)
    y_min = center_y - (label_height/2)
    x_max = center_x + (label_width/2)
    y_max = center_y + (label_height/2)
    
    return np.array([x_min, y_min, x_max, y_max]).T

def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes

def create_val_gt(val_path, output_path, height, width):
    for folder in natsorted(glob.glob(os.path.join(val_path, '*'))):
        print(folder)
        df = pd.DataFrame()
        frame_id = 1
        
        for f in natsorted(glob.glob(os.path.join(folder, 'labels_ftid', '*.txt'))):
            print(f)
            temp = pd.read_csv(
                f, names=['track_id', 'class', 'x', 'y', 'w', 'h'], sep=' ')
            bbox = temp.iloc[:, 2:].to_numpy()
            bbox = ccwh2xyxy(height, width, bbox)
            bbox = xyxy2xywh(bbox)
            temp.iloc[:, 2:] = bbox
            temp['frame_id'] = frame_id
            temp['track_id'] = temp['track_id'].map(str_to_int)
            df = pd.concat([df, temp], axis=0)
            frame_id = frame_id + 1

        df['conf'] = 1
        df['a'] = -1
        df['b'] = -1
        df['c'] = -1
        df = df[['frame_id', 'track_id', 'x',
                 'y', 'w', 'h', 'conf', 'a', 'b', 'c']]
        df = df.sort_values(['track_id', 'frame_id'], ascending=True)
        df.to_csv(os.path.join(output_path, '{}.txt'.format(
            folder.split('/')[-1])), index=False, sep=',', header=False)

def create_annotations(DATA_PATH, OUT_PATH):
    H, W = 0, 0
    
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    data_path = DATA_PATH#os.path.join(DATA_PATH, 'Test')
    out_path = os.path.join(OUT_PATH, 'Test.json')
    out = {'images': [], 'annotations': [], 'videos': [],
            'categories': [{'id': 0, 'name': 'sperm'},
                            {'id': 1, 'name': 'cluster'},
                            {'id': 2, 'name': 'small or pinhead'}]}
    # seqs = os.listdir(data_path)
    seqs = [str(x) for x in Path(data_path).iterdir() if x.is_dir()]
    image_cnt = 0
    ann_cnt = 0
    video_cnt = 0
    tid_curr = 0
    tid_last = -1

    for seq in natsorted(seqs):
        if '.DS_Store' in seq:
            continue
        if 'mot' in DATA_PATH and (split != 'test' and not ('FRCNN' in seq)):
            continue
        if seq.endswith('.ipynb_checkpoints'):
            continue
        video_cnt += 1  # video sequence number.
        out['videos'].append({'id': video_cnt, 'file_name': seq})
        seq_path = os.path.join(data_path, seq)
        img_path = os.path.join(seq_path, 'images')
        ann_path = os.path.join(seq_path, 'labels_ftid')
        images = os.listdir(img_path)
        # half and half
        num_images = len([image for image in images if 'jpg' in image])
        image_range = [0, num_images - 1]
        print('{}: {} images'.format(seq, num_images))

        for i, img_name in enumerate(natsorted(images)):
            if i < image_range[0] or i > image_range[1]:
                continue
            # Image
            img = cv2.imread(os.path.join(
                img_path, img_name))
            print('img path:', os.path.join(
                img_path, img_name))
            
            if H == 0 and W == 0:
                H, W = img.shape[:2]
                
            height, width = img.shape[:2]
            image_info = {'file_name': os.path.join(img_path, img_name),  # image name.
                            # image number in the entire training set.
                            'id': image_cnt + i + 1,
                            # image number in the video sequence, starting from 1.
                            'frame_id': i + 1 - image_range[0],
                            # image number in the entire training set.
                            'prev_image_id': image_cnt + i if i > 0 else -1,
                            'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                            'video_id': seq,  # video_cnt
                            'height': height, 'width': width}
            out['images'].append(image_info)

            # Label
            annotations_path = os.path.join(
                ann_path, '{}_with_ftid.txt'.format(img_name[:-4]))
            track_id_dict = {}
            track_id_count = 1
            if os.path.exists(annotations_path) == True:
                anns = np.genfromtxt(annotations_path, dtype='str')
                anns = np.atleast_2d(anns)
                print('label path:', annotations_path)
                for j in range(anns.shape[0]):
                    frame_id = int(img_name[:-4].split('_')[-1])
                    if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                        continue
                    track_id_str = anns[j][0]
                    # Hashing track_id
                    if track_id_str in track_id_dict.keys():
                        track_id = track_id_dict[track_id_str]
                    else:
                        track_id_dict[track_id_str] = track_id_count
                        track_id = track_id_count
                        track_id_count += 1
                    ###
                    category_id = int(anns[j][1])
                    bbox = np.array(anns[j][2:], dtype='float64').tolist()

                    ### Convert yolo bbox to coco format ###
                    bbox = ccwh2xyxy_single(height, width, bbox)
                    bbox = xyxy2xywh_single(bbox)
                    ### Convert yolo bbox to coco format ###
                    area = np.float64(anns[j][-1]) * \
                        np.float64(anns[j][-2])

                    ann_cnt += 1
                    if not track_id == tid_last:
                        tid_curr += 1
                        tid_last = track_id
                    ann = {
                        'id': ann_cnt,
                        'category_id': category_id,
                        'image_id': image_cnt + frame_id,
                        'track_id': track_id,
                        'bbox': bbox,
                        'conf': 1,
                        'iscrowd': 0,
                        'area': area,
                    }
                    out['annotations'].append(ann)
        image_cnt += num_images
        print(tid_curr, tid_last)

    print('loaded {} images and {} annotations'.format(len(out['images']), len(out['annotations'])))
    json.dump(out, open(out_path, 'w'))
    return H, W

def merge_videos(videos, save_path):
    
    print("Merging videos")
    # Create a new video
    vid_cap = cv2.VideoCapture(videos[0])
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(fps, w, h)
    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    # Write all the frames sequentially to the new video
    for v in videos:
        curr_v = cv2.VideoCapture(v)
        while curr_v.isOpened():
            r, frame = curr_v.read()    # Get return value and curr frame of curr video
            if not r:
                break
            video.write(frame)          # Write the frame

    video.release() 
    print("Saved to ", save_path)