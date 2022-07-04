from PIL import Image
import os

for dirPath, dirNames, fileNames in os.walk("/data1/1K_New/train_binary_mask_by_DRFI/"):
    for f in fileNames:
        p = os.path.join(dirPath, f)
        print(p)
        im = Image.open(p)
        im.save(p.replace("JPEG", "png"))

