# referrence : https://github.com/ultralytics/yolov5/issues/36
#%% 
import torch
import cv2
from IPython.display import display
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image

import io
import numpy as np
#%% custom Model loading 
model = torch.hub.load(
   '../yolov5', # 저장소 위치
   'custom' , # 커스텀 웨이트 파일 사용
   './weights/yolov5s.pt',
   source='local' # 로컬 저장소 사용
   )
#    pretrained=True)

#%% Images
model.conf = 0.5  # NMS confidence threshold
model.iou = 0.2  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
model.max_det = 1000  # maximum number of detections per image
model.amp = False  # Automatic Mixed Precision (AMP) inference

# Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

#%% Inference files
imgs = ['./bus.jpg']  # batch of images

#%% Inference from files
results = model(imgs)

#%% Inference from opencv 
results = model(cv2.imread(imgs[0])[:,:,::-1])

#%% Inference from PIL
with open(imgs[0], "rb") as fd:
    img_data = fd.read()
    np_img = np.array(Image.open(io.BytesIO(img_data))) # PIL을 사용한 이미지 로딩
    results = model(np_img)


#%%
# Results
results.print()
results.save()  # or .show() 결과 이미지 저장

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)

#%% json results string
result_json = results.pandas().xyxy[0].to_json(orient="records") 
print(result_json)

#%% cv2 로 이미지 처리 
img = cv2.imread(imgs[0])  # BGR -> RGB
result_img = img.copy()
dfResult = results.pandas().xyxy[0]
# row = dfResult.iloc[0]
print(dfResult.to_numpy())

for det in dfResult.to_numpy():
    x1,y1,x2,y2 = det[:4]
    cv2.rectangle(result_img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
display(Image.fromarray(cv2.cvtColor(result_img,cv2.COLOR_BGR2RGB)))


#%% pillow 로 처리하기 
# results.xyxy[0]
with open(imgs[0], "rb") as fd:
    img_data = fd.read()
    np_img = np.array(Image.open(io.BytesIO(img_data)).copy()) # PIL을 사용한 이미지 로딩
    _img = Image.fromarray(np_img)
    drawer = ImageDraw.Draw(_img)
    
    for i, pred in enumerate(results.xyxy[0]):
        # print(pred,i)
        x1,y1,x2,y2 = pred[:4]
        # print(int(x1),int(y1),x2,y2)
        drawer.rectangle(
                [int(x1),int(y1),int(x2),int(y2)],
                fill=None,
                outline="rgb(255,255,0)",
                width=2
            )
    display(_img)
        
    

# %%
