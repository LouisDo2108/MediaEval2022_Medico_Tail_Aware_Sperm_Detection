### Clone yolov5
# !git clone https://github.com/ultralytics/yolov5
# !pip install -r ./yolov5/requirements.txt
# %cd yolov5

### Setup kaggle
# ! pip install -q kaggle
# from google.colab import files
# files.upload()         # expire any previous token(s) and upload recreated token
# ! cp kaggle.json ~/.kaggle/
# ! chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d vlbthambawita/visemtracking

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

DATA_PATH = "../input/visemtracking/VISEM_Tracking_Train_v4/Train"

num = np.sort(np.array(os.listdir(DATA_PATH), dtype='int32'))

from sklearn.model_selection import train_test_split

train_num, val_num = train_test_split(num, test_size=0.2, random_state=42)

# Prepare a file which contain train and validation data information and class information
main_train_dir = "/kaggle/input/visemtracking/VISEM_Tracking_Train_v4/Train"

with open("./yolov5/train_val.yaml", "w") as f:

    # Writing train paths to yaml
    f.write("train: [ \n")
    for t in train_num:
        f.write(main_train_dir + f"/{t}," + "\n")
    f.write("]\n\n")

    # writing validation paths to yaml
    f.write("val: [\n")
    for v in val_num:
        f.write(main_train_dir + f"/{v}," + "\n")
    f.write("]\n\n")

    # writing number of class parameter
    f.write("nc: 3\n\n")

    # Writing class names
    f.write('names: [ "sperm", "cluster", "small_or_pinhead"]')
f.close()

### Train yolov5
# !python ./yolov5/train.py --img 640 --batch 16 --epochs 100\
# --data ./yolov5/train_val.yaml --weights yolov5s.pt\
# --cache disk