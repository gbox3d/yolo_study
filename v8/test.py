#%%
import cv2

from IPython.display import display
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image

# import ultralytics
from ultralytics import YOLO,checks

checks()
# %%
model = YOLO('/home/ubiqos-ai2/work/visionApp/yolo_study/datasets/wbd/runs/detect/train4/weights/best.pt')

#%%
im = cv2.imread('/home/ubiqos-ai2/work/visionApp/yolo_study/datasets/wbd/train/images/bank_5_1622018550499_jpg.rf.26a74d33ec1378137c2108faec8d6021.jpg')
results = model(source=im,conf=0.5)  # predict on an image

# %%
result_img = im.copy()

for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cv2.rectangle(result_img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        print(box.xyxy)
        print(box.xywh)
        print(box.cls)
        print(box.conf)
display(Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)))
#%%