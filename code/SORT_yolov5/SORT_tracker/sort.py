import numpy as np
import pandas as pd

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
from kalman_filter import KalmanFilterNew as KalmanFilter
from association import associate_detections_to_trackers
from estimation.speed import TWOLINEs


def onSegment(p, q, r):
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and 
           (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        return True
    return False
  
def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
      
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/ 
    # for details of below formula. 
      
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
    if (val > 0):
          
        # Clockwise orientation
        return 1
    elif (val < 0):
          
        # Counterclockwise orientation
        return 2
    else:
          
        # Collinear orientation
        return 0
  
# The main function that returns true if 
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1, q1, p2, q2):
      
    # Find the 4 orientations required for 
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
  
    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True
  
    # Special Cases
  
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True
  
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True
  
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True
  
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True
  
    # If none of the cases
    return False


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x,score=None, class_id = None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None) or (class_id == None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score, class_id], dtype=object).reshape((1,6))

class OUTPUT():
  def __init__(self, bbox, conf, class_id, track_id, speed):
    self.bbox = bbox
    self.conf = conf
    self.class_id = class_id
    self.track_id = track_id
    self.speed = speed


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox, score, class_id):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.class_id = class_id
    self.conf_score = score

    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

    self.speed = 0
    self.cached_pos = None
    self.cached_lines = []
    self.start_t = 0
    self.counted = False

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x, self.conf_score, self.class_id))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x, self.conf_score, self.class_id)



class SORT(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.1, speedlines=None, countline = None):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0
    self.model_estimation = TWOLINEs(9,speedlines=speedlines)
    
    self.cars = 0
    self.motors = 0
    self.countline = countline

  def update(self, frame_idx, dets=np.empty((0, 6))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score, class_id],[x1,y1,x2,y2,score, class_id],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 6)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(pd.isna(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:4], dets[i, 4], dets[i, 5])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
      d = trk.get_state()[0]
      x1, y1, x2, y2 = d[:4]
      xmean = x1 + (x2 - x1)/2
      ymean = y1 + (y2 - y1)/2

      if trk.cached_pos == None:
        trk.cached_pos = [xmean, ymean]
      else:
        nline = [trk.cached_pos, [xmean,y2]]
         
        if doIntersect(self.countline[0], self.countline[1], nline[0], nline[1]):
          if not trk.counted:
              trk.counted = True
              if trk.class_id == 3:
                self.cars += 1
              if trk.class_id == 4:
                self.motors += 1

        for line in self.model_estimation.speedlines:
          if doIntersect(line[0], line[1], nline[0], nline[1]):
            if line not in trk.cached_lines:
              trk.cached_lines.append(line)
              if len(trk.cached_lines) > 1:
                trk.speed = self.model_estimation.estimate_speed(trk, frame_idx, self._tlwh_to_xywh([x1, y1, x2, y2]))
              trk.start_t = frame_idx
              trk.start_pos = self._tlwh_to_xywh([x1,y1,x2,y2])
              
      trk.cached_pos = [xmean, ymean]
      if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
        output = OUTPUT(d[:4],d[4], d[5], trk.id+1, trk.speed)
        ret.append(output)
      i -= 1
      # remove dead tracklet
      if(trk.time_since_update > self.max_age):
        self.trackers.pop(i)
    if(len(ret)>0):
      return np.array(ret)
    return np.empty((0,6))

  def _tlwh_to_xywh(self, bbox_tlwh):
    x,y,w,h= bbox_tlwh
    x1 = int(x + w/2)
    y1 = int(y + h/2)
    return x1, y1, w, h 

