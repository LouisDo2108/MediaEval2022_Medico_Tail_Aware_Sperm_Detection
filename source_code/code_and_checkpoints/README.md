# YoloV7 trained models:
https://drive.google.com/drive/folders/10oE7MkhagPhJPCdZuMUCezbnvBNCpo2T?usp=share_link
# Visem dataset:
https://drive.google.com/drive/folders/17JWZ_SJ9fvFpKTPUXlkjrJIy8DuooEoi?usp=share_link

# Setup
1. Create a conda environemnt with python==3.8, pip3 install -r requirements.txt
2. set current working directory as  /source_code/code_and_checkpoints/

# Run
## run_detect.sh
Sperms detection using YoloV7 \
Parameters: \
 --img-size: the input image size of the yolo model \
--source: the .mp4 videos \
--weights: the yolo model .pt weight \
--suppress: a flag to activate border suppression \

## run_track.sh
Track sperms in a video using SORT \
Parameters: \
--img-size: the input image size of the yolo model \
--source: the .mp4 videos \
--yolo-weights: the yolo model .pt weight \
--suppress: a flag to activate border suppression
