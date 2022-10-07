import os
from YoloV5_utils.utils import create_yaml

if __name__ == "__main__":
    
    main_train_dir = "D:\\VISEM_Tracking_Train_v4\\"
    yaml_path = create_yaml(main_train_dir)

    command = f"cd yolov5/ &\
        python3 train.py\
        -img 640 \
        --batch 16\
        --epochs 3 \
        --data {yaml_path}\
        --weights yolov5s.pt"

    os.system(command)