# %%
# init module
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn


import numpy as np
import torch.nn as nn

from numpy import random

from IPython.display import display
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image

import sys
sys.path.append('../libs')

from yolov5_myutil import initModel,predict
from utils.torch_utils import select_device
from utils.general import scale_coords


print('extention load complete')
# %%
# 연산장치 얻기 cpu 인지 gpu 인지 판단
device = select_device()
print(device.type)

# %%
# load model
model,imgsz = initModel(640,device,'yolov5s.pt')


# %%
# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
# colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
print(names)
# print(colors)
# %%
# init predict

img0 = cv2.imread('./bus.jpg')  # BGR
pred,_img = predict(model,img0,device,imgsz)

print(img0.shape)
print(_img.shape)

# %%
result_img = img0.copy()
for i, det in enumerate(pred):
    # print(det)
    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_coords(_img.shape[2:], det[:, :4], img0.shape).round()

    for *xyxy, conf, cls in reversed(det):
        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        print('%12s %.2f %.2f %.2f %.2f %.2f' % (names[int(cls)], conf,c1[0],c1[1],c2[0],c2[1]))
        # print(c1,c2)
        cv2.rectangle(result_img, c1, c2, (0,0,255), thickness=3, lineType=cv2.LINE_AA)
        
        
    # Print results
    for c in det[:, -1].unique():
        n = (det[:, -1] == c).sum()  # detections per class
        print('%g %ss, ' % (n, names[int(c)]))

display(Image.fromarray(cv2.cvtColor(result_img,cv2.COLOR_BGR2RGB)))

# %%
