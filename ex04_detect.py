# %%
# init module
import time
import io
import numpy as np
from pathlib import Path

import cv2

from IPython.display import display
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image

import sys

from modules.yl5Detector import yl5Detector

print('extention load complete')
#%%
color_table = [(0,0,0),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255)]
#%%
weight_path = '/home/gbox3d/work/study/yolo_study/weights/exp05/last.pt'
_detectorObj = yl5Detector(weight_path,640,'')
detector_device = _detectorObj.device
detector_device_type = str(_detectorObj.device)
detector_label_names = _detectorObj.names
detector_imgsz = _detectorObj.imgsz

print('setup yolov5 pytorch')
#%% cv2로 이미지 처리하기 
img = cv2.imread('./d_05.jpg')  # BGR
np_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
pred,_img = _detectorObj.predict(
    np_img,
    scaling=True,
    _conf_thres=0.25,
    colorFormat='rgb'
)

result_img = img.copy()
for i, det in enumerate(pred):
    print(det)
    for *xyxy, conf, cls in reversed(det):
        
        # print(conf, detector_label_names[int(cls)],cls)
        print(cls)
        
        cv2.rectangle(result_img, xyxy[0], xyxy[1], 
            color_table[cls % len(color_table) ],  # color
            thickness=1, 
            lineType=cv2.LINE_AA)
        cv2.putText(result_img, f'{cls}', xyxy[0],
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_table[cls % len(color_table) ], 1, cv2.LINE_AA)

display(Image.fromarray(cv2.cvtColor(result_img,cv2.COLOR_BGR2RGB)))


# %%
