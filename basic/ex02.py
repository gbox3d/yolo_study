# %%
# init module

from IPython.display import display
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image


import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn


import numpy as np
import torch.nn as nn

from numpy import random


import sys
sys.path.append('../../yolov5')

from models.common import Conv, DWConv
from models.experimental import attempt_load, Ensemble
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


print('extention load complete')
# %%
# 연산장치 얻기 cpu 인지 gpu 인지 판단
device = select_device()
print(device.type)

# %%
# cpu가 아니라면
half = device.type != 'cpu'
print(half)
# %%
# load model


def load_model(weights, imgsz, map_location):
    model = Ensemble()
    model.append(torch.load(weights, map_location=map_location)[
                 'model'].float().fuse().eval())  # load FP32 model

    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


imgsz = 640
map_location = device
model = load_model('yolov5s.pt', imgsz, map_location)

imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
print(f'image size {imgsz}')
if half:
    print('conver fp16')
    model.half()  # to FP16

print('model load done')


# %%
# load image

img0 = cv2.imread('./bus.jpg')  # BGR
img, _ratio, _dsize = letterbox(img0, new_shape=imgsz)
print(_dsize, _ratio)
# Convert
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
img = np.ascontiguousarray(img)

type(img)

# %%
# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
print(names)
print(colors)
# %%
# init predict
_img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
# run once
_ = model(_img.half() if half else _img) if device.type != 'cpu' else None


# %%

w_img = img.copy()

w_img = torch.from_numpy(w_img).to(device)
w_img = w_img.half() if half else w_img.float()  # uint8 to fp16/32
w_img /= 255.0  # 0 - 255 to 0.0 - 1.0
if w_img.ndimension() == 3:
    w_img = w_img.unsqueeze(0)

# Inference
# t1 = time_synchronized()
pred = model(w_img, augment=False)[0]

# Apply NMS
pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

print(pred)
# t2 = time_synchronized()
# %%
for i, det in enumerate(pred):
    print(det)
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]] 
    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_coords(w_img.shape[2:], det[:, :4], img0.shape).round()

    for *xyxy, conf, cls in reversed(det):
        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        # line = (cls, *xywh, conf)
        # print(line)
        # print( ('%g ' * len(line)).rstrip() % line + '\n' )
        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        print('%s %.2f' % (names[int(cls)], conf))
        print(c1,c2)
        
        
    # Print results
    for c in det[:, -1].unique():
        n = (det[:, -1] == c).sum()  # detections per class
        print('%g %ss, ' % (n, names[int(c)]))

# %%
