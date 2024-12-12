#%%
import cv2 as cv
import numpy as np

from IPython.display import display
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image

from ultralytics import YOLO

#%%
img = cv.imread("../bus.jpg")
# model = YOLO("yolov8n-seg.pt")
model = YOLO("yolo11n-seg.pt")  
results = model.predict(img, conf=0.5, iou=0.7, classes=None)

#%%
print(results)
print(len(results))

result = results[0]
print(result.names)

#%% box 정보
_img = result.orig_img.copy()
for box in result.boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    print(f'min: {(x1.item(),y1.item())} max: {(x2.item(),y2.item())}')
    
    x,y,w,h = box.xywh[0]
    print(f'x: {x.item()} y: {y.item()} w: {w.item()} h: {h.item()}')
    
    cv.rectangle(_img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
    
    print(f'class : {int(box.cls)}')
    print(f'conf : {box.conf.item()}')
    
display(Image.fromarray(cv.cvtColor(_img, cv.COLOR_BGR2RGB)))


# %%
print( result.speed)

# %%
print(result.masks)
print(result.masks.data)

# %% mask 정보
for _mask_data in result.masks.data:
    mask_img = _mask_data.cpu().numpy()
    mask_img = np.where(mask_img > 0.5, 255, 0).astype('uint8')  # 0.5 이상인 값을 255로 바꾸고, uint8로 변경
    display(Image.fromarray(cv.cvtColor(mask_img, cv.COLOR_BGR2RGB)))
    
# %% segment 정보
_seg_img = result.orig_img.copy()
img_h, img_w, _ = _seg_img.shape

for _segment in result.masks.xyn:  # 수정된 부분
    # 좌표값을 이미지 크기에 맞게 정수형으로 변환
    np_cnt = (_segment * [img_w,img_h]).astype(np.int32)    
    cv.polylines(_seg_img, [np_cnt], True, (0, 255, 0), 2)

display(Image.fromarray(cv.cvtColor(_seg_img, cv.COLOR_BGR2RGB)))
    
# %%

print(result.masks.xy)

# %%
_seg_img = result.orig_img.copy()
img_h, img_w, _ = _seg_img.shape

# 각 박스에 대해 반복
for idx, box in enumerate(result.boxes):
    # 클래스 이름이 'person'인 경우만 윤곽선을 그립니다.
    if result.names[int(box.cls)] == 'person':
        
        _segment = result.masks.xyn[idx]  # 이 박스에 해당하는 세그먼트를 가져옵니다. (노멀라이즈된 좌표값)
        # 좌표값을 이미지 크기에 맞게 정수형으로 변환
        np_cnt = (_segment * [img_w,img_h]).astype(np.int32)    
        
        
        cv.polylines(_seg_img, [np_cnt], True, (0, 255, 0), 2)
        
        # 박스를 빨간색으로 그립니다.
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv.rectangle(_seg_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색으로 박스를 그립니다.

display(Image.fromarray(cv.cvtColor(_seg_img, cv.COLOR_BGR2RGB)))

# %%
