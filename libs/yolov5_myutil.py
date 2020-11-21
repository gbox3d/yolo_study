import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn


import numpy as np
import torch.nn as nn

from numpy import random

from models.common import Conv, DWConv
from models.experimental import  Ensemble
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
# from utils.plots import plot_one_box
# from utils.torch_utils import select_device

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

def initModel(imgsz,map_location,weights_path) :
    # imgsz = 640
    # map_location = device
    model = load_model(weights_path, imgsz, map_location)

    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    print(f'image size {imgsz}')
    if map_location.type != 'cpu': # gpu 라면 
        print(f'{map_location.type} device convert fp16')
        model.half()  # to FP16

    print('model load done')
    return (model,imgsz)

# init predict
def predict(model,np_img,device,imgsz) :
    half = device.type != 'cpu'
    img, _ratio, _dsize = letterbox(np_img, new_shape=imgsz)
    # print(_dsize, _ratio)
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    _img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(_img.half() if half else _img) if device.type != 'cpu' else None

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
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    return (pred,img)
    # print(pred)
