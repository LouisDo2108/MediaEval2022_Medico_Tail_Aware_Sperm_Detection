import os
from natsort import natsorted
import cv2
import numpy as np
import json
from ByteTrack_utils.utils.convert_bbox_format import ccwh2xyxy_single, xyxy2xywh_single


def create_annotations(DATA_PATH, OUT_PATH, SPLITS=['Train', 'Val']):
    H, W = 0, 0
    
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        if split == "Val":
            data_path = os.path.join(DATA_PATH, 'Val')
        else:
            data_path = os.path.join(DATA_PATH, 'Train')
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
        out = {'images': [], 'annotations': [], 'videos': [],
               'categories': [{'id': 0, 'name': 'sperm'},
                              {'id': 1, 'name': 'cluster'},
                              {'id': 2, 'name': 'small or pinhead'}]}
        seqs = os.listdir(data_path)
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

        print('loaded {} for {} images and {} annotations'.format(
            split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))
    return H, W
