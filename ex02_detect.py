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

# sys.path.append('../libs')

# from yolov5_myutil import createModel,predict
from modules.yl5Detector import yl5Detector

print('extention load complete')
#%%
weight_path = './yolov5s.pt'
_detectorObj = yl5Detector(weight_path,640,'',logging=False)
detector_device = _detectorObj.device
detector_device_type = str(_detectorObj.device)
detector_label_names = _detectorObj.names
detector_imgsz = _detectorObj.imgsz

print('setup yolov5 pytorch')


#%% cv2로 이미지 처리하기 
img = cv2.imread('./bus.jpg')  # BGR
np_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
pred,_img = _detectorObj.predict(
    np_img,
    scaling=True,
    colorFormat='rgb'
)

result_img = img.copy()
for i, det in enumerate(pred):
    print(det)
    for *xyxy, conf, cls in reversed(det):
        
        print(conf, detector_label_names[int(cls)],cls)
        
        cv2.rectangle(result_img, xyxy[0], xyxy[1], 
            (0,0,255),  # color
            thickness=3, 
            lineType=cv2.LINE_AA)

display(Image.fromarray(cv2.cvtColor(result_img,cv2.COLOR_BGR2RGB)))

#%% pil 로 이미지 처리하기 
with open("./bus.jpg", "rb") as fd:
    img_data = fd.read()

    img0 = Image.open(io.BytesIO(img_data))  # PIL을 사용한 이미지 로딩
    np_img = np.array(img0.copy())

    print(np_img.shape)
        
    _pred,_img  = _detectorObj.predict(np_img,
            scaling=True,
            colorFormat='rgb')
    print(_img.shape)

    _img = Image.fromarray(np_img)
    drawer = ImageDraw.Draw(_img)

    for __det in _pred[0] :
            # __det = _r[0][0]
            print(__det)
            drawer.rectangle(
                [__det[0],__det[1]],
                fill=None,outline="red"
            )
    display(_img)

# %%
