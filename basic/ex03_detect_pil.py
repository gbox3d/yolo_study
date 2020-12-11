# PIL 로 이미지를 읽어 들여 감지 하는 예제 
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
model,imgsz = initModel(640,device,'../models/yolov5s.pt')


# %%
# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
# colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
print(names)
# print(colors)
# %%
# init predict
img0 = Image.open('./bus.jpg') # PIL을 사용한 이미지 로딩 
np_img = np.array(img0.copy())
print(f'orginal size {np_img.shape}')
pred,_img = predict(model,np_img,device,imgsz,scaling=True,colorFormat='rgb')
print(f'resize  {_img.shape}')

# %%
result_img = img0.copy()
for i, det in enumerate(pred):

    for *xyxy, conf, cls in reversed(det):
        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        print('%12s %.2f %.2f %.2f %.2f %.2f' % (names[int(cls)], conf,c1[0],c1[1],c2[0],c2[1]))

        left = c1[0]
        top = c1[1]
        right = c2[0]
        bottom = c2[1]
        
        drawer = ImageDraw.Draw(result_img)
        drawer.line([(left, top), (left, bottom), (right, bottom), (right, top),
                   (left, top)],
                  width=4,
                  fill=(255,0,0))
        
    # Print results
    for c in det[:, -1].unique():
        n = (det[:, -1] == c).sum()  # detections per class
        print('%g %ss, ' % (n, names[int(c)]))

# display(Image.fromarray(cv2.cvtColor(result_img,cv2.COLOR_BGR2RGB)))
display(result_img)
