import math
import numpy as np

class TWOLINEs(object):

    def __init__(self, fps=24, ppm=192, speedlines=None):
        self.fps = fps
        self.ppm = ppm
        self.speedlines = speedlines
        
    def estimate_speed(self, obj, frame_idx, pos):
        if obj.start_pos is not None:
            # d_pixels = math.sqrt(math.pow(pos[0] - obj.start_pos[0], 2) + math.pow(pos[1] - obj.start_pos[1], 2))
            # d_meters = d_pixels / self.ppm
            # speed = d_meters * self.fps * 3.6
            time_travel = (frame_idx - obj.start_t) / self.fps
            distance = 30
            if time_travel == 0:
                return 0
            speed = (distance / time_travel) * 3.6
            return speed
        return 0
