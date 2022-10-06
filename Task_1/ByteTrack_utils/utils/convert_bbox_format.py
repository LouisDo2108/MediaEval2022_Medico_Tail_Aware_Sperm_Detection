import numpy as np

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