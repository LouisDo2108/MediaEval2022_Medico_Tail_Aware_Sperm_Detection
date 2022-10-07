import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path

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