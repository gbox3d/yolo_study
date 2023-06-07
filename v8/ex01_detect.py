#%%
import cv2 as cv

from IPython.display import display
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image

# import ultralytics
from ultralytics import YOLO,checks

checks()
# %%
model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model
#%%

model.info()
model.names

#%%
im = cv.imread('./bus.jpg')
results = model(source=im,conf=0.5)  # predict on an image

# %%
result_img = im.copy()

for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cv.rectangle(result_img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        print(box.xyxy)
        print(box.xywh)
        print(f'class : {int(box.cls.cpu().item())}' )
        print(f'conf : {int(box.conf.cpu().item()*100)}%' )
display(Image.fromarray(cv.cvtColor(result_img, cv.COLOR_BGR2RGB)))
#%%
_boxes = results[0].boxes.data.cpu().detach().numpy()

for box in _boxes:
    print(box)
    
    
# %%
