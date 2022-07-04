import cv2
import numpy as np
import os
path = r"/data1/1K_New/train_binary_mask_by_DRFI"
new_path = r"/data1/1K_New/train_binary_mask_by_DRFI_npz"
from glob import glob
class_dirs = glob(path + "/*/", recursive = True)
# print(len(class_dirs))

for class_dir in class_dirs:
    class_dir_temp = class_dir.replace("\\","/")
    class_name = class_dir_temp.split("/")[-2]
    print(class_name)
    if not os.path.isdir(os.path.join(new_path,class_name)):
        os.makedirs(os.path.join(new_path,class_name))
    for file in os.listdir(class_dir):
        mask = cv2.imread(os.path.join(class_dir,file), cv2.IMREAD_GRAYSCALE)
        mask_f = mask.astype(bool)
        np.savez_compressed(os.path.join(new_path,class_name,file.split(".")[0]), foreground=mask_f)
        #
        # mask_file = np.load(r"D:\MPLCRL\n07875152_2309.npz")