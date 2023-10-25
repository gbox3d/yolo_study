#%%
import cv2 as cv
import numpy as np

from IPython.display import display
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image

from ultralytics import YOLO
from ultralytics.engine.results import Results

#%%
try :
    img = cv.imread("image/nanace1.jpg")
    model = YOLO("yolov8l-seg.pt")  
    print("load model ok")
    print(model.names)
except Exception as e:
    print(e)
    print("load model fail")
#%%
results = model.predict(img, conf=0.5, iou=0.7, classes=None,device='cpu')
print(f'len(results): {len(results)}')
result = results[0]


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
print(result.masks)
print(result.masks.data)

# %% mask 정보
# for _mask_data in result.masks.data:
_mask_data = result.masks.data[0]
mask_img = _mask_data.cpu().numpy()
mask_img = np.where(mask_img > 0.5, 255, 0).astype('uint8')  # 0.5 이상인 값을 255로 바꾸고, uint8로 변경
display(Image.fromarray(cv.cvtColor(mask_img, cv.COLOR_BGR2RGB)))
#%%
alpha_channel = mask_img  # 알파 채널로 이진 마스크 사용
_, _, channels = img.shape
if channels < 4:
    rgba_mask = cv.cvtColor(mask_img, cv.COLOR_GRAY2RGBA)  # GRAY to RGBA
    rgba_mask[:, :, 3] = alpha_channel  # 알파 채널 설정
else:
    rgba_mask = mask_img.copy()
    rgba_mask[:, :, 3] = alpha_channel  # 알파 채널 설정
#%%

# 원본 이미지를 RGBA로 변환
_, _, channels = img.shape
if channels < 4:
    rgba_image = cv.cvtColor(img, cv.COLOR_BGR2RGBA)
else:
    rgba_image = img.copy()

# %%
print(rgba_image.shape)  # 원본 이미지의 차원 출력
print(rgba_mask.shape)   # 마스크 이미지의 차원 출력
print(rgba_image.dtype)  # 원본 이미지의 데이터 유형 출력
print(rgba_mask.dtype)   # 마스크 이미지의 데이터 유형 출력

#%%
# 원본 이미지의 차원에 맞게 마스크 이미지 리사이즈
rgba_mask_resized = cv.resize(rgba_mask, (rgba_image.shape[1], rgba_image.shape[0]), interpolation=cv.INTER_AREA)

# 리사이즈된 마스크 이미지와 원본 이미지 합성
masked_image = cv.bitwise_and(rgba_image, rgba_mask_resized)

# 결과 출력
display(Image.fromarray(cv.cvtColor(masked_image, cv.COLOR_RGBA2RGB)))

# %%
