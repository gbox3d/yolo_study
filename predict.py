#%%
import argparse
import os
import sys
from pathlib import Path

import cv2
from numpy import source
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
# %%
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

#%%
device = 0
weights= './yolov5s.pt'
imgsz=(640, 640)
data='/home/gbox3d/work/visionApp/yolov5/data/coco128.yaml'
dnn=False  # use OpenCV DNN for ONNX inference
half=False

device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size(imgsz, s=stride)  # check image size

#%%
# Half
half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
if pt or jit:
    model.model.half() if half else model.model.float()
    
    
#%%
source = '/home/gbox3d/work/visionApp/yolov5/data/images/bus.jpg'
dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

bs = 1  # batch_size
model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup



# %%
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS
conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000  # maximum detections per image

for path, im, im0s, vid_cap, s in dataset:
    # print(im)
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    pred = model(im,augment=False)
    # print(pred)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            print(s)
            
            for *xyxy, conf, cls in reversed(det):
                xmin = int(xyxy[0])
                ymin = int(xyxy[1])
                xmax = int(xyxy[2])
                ymax = int(xyxy[3])
                
                print(f'min: {xmin}, {ymin}, max: {xmax}, {ymax} , conf: {conf}, class: {int(cls)}')
        
    

# %%
