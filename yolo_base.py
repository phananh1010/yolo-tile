import yolo_base
from yolo.utils import utils
from yolo import models

import torch as t
import imp
imp.reload(models)

YOLOHOME='/home/anh2/workspace/test/yolo'
model_def=YOLOHOME + '/' + "config/yolov3-tiny-base.cfg"#"config/yolov3-base.cfg"#
weights_path=YOLOHOME + '/' + "weights/yolov3-tiny.weights"#"weights/yolov3.weights"#
#img_size=601#1920#416
image_folder = YOLOHOME + '/' + "data/samples"
class_path=YOLOHOME + '/' + 'data/coco.names'
classes = utils.load_classes(class_path)


def load_model(img_size):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model = models.Darknet(model_def, img_size=img_size).to(device)
    model.load_darknet_weights(weights_path)
    model.eval()
    return model
    

