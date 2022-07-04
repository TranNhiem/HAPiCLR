import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, default_argument_parser, default_setup
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


args = default_argument_parser().parse_args()
cfg = setup(args)
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
im = cv2.imread("/data1/dataset/VOC2007/JPEGImages/009703.jpg")
# cv2_imshow(im)

predictor = DefaultPredictor(cfg)
outputs = predictor(im)
output_file_name = "./output.jpg"
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# out = v.draw_box(outputs["instances"].pred_boxes.to("cpu"))
cv2.imwrite(output_file_name, out.get_image()[:, :, ::-1])
print("result is save in",output_file_name)