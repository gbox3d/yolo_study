#%%
# import socket
import io
# from struct import *

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from numpy import random

from IPython.display import display
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image

import sys

# sys.path.append("./")
sys.path.append("./modules/yolov5")


from modules.yolov5.utils.datasets import letterbox
from modules.yolov5.models.experimental import attempt_load
from modules.yolov5.utils.general import check_img_size,non_max_suppression,scale_coords,set_logging
from modules.yolov5.utils.torch_utils import select_device

class yl5Detector:
    def __init__(self,weights, imgsz , device,logging=True):
        
        # weights, imgsz , device= opt.weights,opt.img_size,opt.device

        if logging == True :
            set_logging()

        device = select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        
        self.model = model
        self.imgsz = imgsz
        self.device = device
        self.names = names
        self.stride = stride
        self.half = half
    def predict(self,np_img, imgsz=640, scaling=False, colorFormat='bgr', _conf_thres=0.25, _iou_thres=0.45, _classes=None, _agnostic=False):

        model = self.model
        half = self.half
        device = self.device
        # half = device.type != 'cpu'
        # print(f'original size : {np_img.shape}')

        if np_img.shape[2] > 3 :
            np_img = np_img[:, :, 0:3]
        
        img, _ratio, _dsize = letterbox(np_img, new_shape=imgsz)
        # print(f'resize : {img.shape}')
        # Convert
        if colorFormat == 'bgr':
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            # img = np.ascontiguousarray(img)
        else:
            # img = img[:, :, ::-1].transpose(2, 0, 1)
            img = img[:, :, ::1].transpose(2, 0, 1)  # RGB 형식 그대로 유지한체로 차원 뒤집기

        img = np.ascontiguousarray(img)
        _img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img

        # # run once
        # _ = model(_img.half() if half else _img) if device.type != 'cpu' else None
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        # w_img = img.copy()

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        # t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=_conf_thres,
                                iou_thres=_iou_thres, classes=_classes, agnostic=_agnostic)

        if scaling:
            for i, det in enumerate(pred):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], np_img.shape).round()

        _pred = [
            [
                (
                    (int(xyxy[0]), int(xyxy[1])),
                    (int(xyxy[2]), int(xyxy[3])),
                    float(conf),
                    int(cls),
                )
                for *xyxy, conf, cls in reversed(det)
            ]
            for det in pred
        ]

        return (_pred, img)

#%%
if __name__ == "__main__":
    
    # opt = {
    #     "weights" : "./weights/yolov5/yolov5s.pt",
    #     "imgsz" : 640,
    #     "device" : ''
    # }
    
    detObj = yl5Detector("./weights/yolov5/yolov5s.pt",640,'')

#%%

    with open("../sample/bus.jpg", "rb") as fd:
        img_data = fd.read()

        img0 = Image.open(io.BytesIO(img_data))  # PIL을 사용한 이미지 로딩
        np_img = np.array(img0.copy())

        print(np_img.shape)
        
        _pred,_img  = detObj.predict(np_img,
            scaling=True,
            colorFormat='rgb')
        print(_img.shape)

        # print(pred)

        # _r = [
        #     [
        #         (
        #             (int(xyxy[0]), int(xyxy[1])),
        #             (int(xyxy[2]), int(xyxy[3])),
        #             float(conf),
        #             int(cls),
        #         )
        #         for *xyxy, conf, cls in reversed(det)
        #     ]
        #     for det in _detections
        # ]
        print(_pred)

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
