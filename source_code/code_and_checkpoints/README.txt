# YoloV7 trained models:
https://drive.google.com/drive/folders/10oE7MkhagPhJPCdZuMUCezbnvBNCpo2T?usp=share_link
# Visem dataset:
https://drive.google.com/drive/folders/17JWZ_SJ9fvFpKTPUXlkjrJIy8DuooEoi?usp=share_link
# Setup
Create a conda environemnt with python==3.8, pip3 install -r requirements.txt

# Run
## run_detect.sh
 --img-size: the input image size of the yolo model 
--source: the .mp4 videos
--weights: the yolo model .pt weight
--suppress: a flag to activate border suppression

## run_track.sh
1. Track sperms in a video. (Optional: --suppress)
--img-size: the input image size of the yolo model 
--source: the .mp4 videos
--yolo-weights: the yolo model .pt weight
--suppress: a flag to activate border suppression