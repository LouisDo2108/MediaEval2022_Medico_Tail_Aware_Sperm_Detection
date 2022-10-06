# MediaEval2022_Medico

## Train trên Yolov5
![](deepsort/yolov5.gif)

## Yolov5 + DeepSort
![](deepsort/yolov5anddeepsort.gif)

## Task 1 (Detection and Tracking evaluations): 
(Consider /content/medico/ as the root directory)
1. Clone https://github.com/LouisDo2108/ByteTrack.git and setup normally by following the readme.md instructions.
```
cd /content/medico/
git clone https://github.com/LouisDo2108/ByteTrack.git &\
cd /content/medico/ByteTrack &\
pip3 install -r requirements.txt &\
python3 setup.py develop &
```
2. Go back to root folder and clone https://github.com/ultralytics/yolov5 and setup YoloV5 normally by following the readme.md instructions.
```
cd /content/medico/ &\
git clone https://github.com/ultralytics/yolov5 &\
cd /content/medico/yolov5 &\
pip3 install -r requirements.txt
```
3. Install the requirements.txt in task_1 folder.
4. Download the visem dataset and unzip it to the root folder.
5. Run Task_1/prepare_data.py. For example:
```
python /content/medico/Task_1/prepare_data.py \
--root_dir /content/medico/VISEM_Tracking_Train_v4/ \
--gt_dir /content/medico/gt/ 
```
6. Run Task_1/evaluate.py
```
python /content/medico/Task_1/evaluate.py \
--result_dir /content/medico/result/ \
--gt_dir /content/medico/gt/ \
--yolo_model_path /content/medico/yolo_trained_models/best.pt
```



