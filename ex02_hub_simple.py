# referrence : https://github.com/ultralytics/yolov5/issues/36
#%% 
import torch

#%%
# custom Model loading 
model = torch.hub.load(
   '../yolov5', # 저장소 위치
   'custom' , # 커스텀 웨이트 파일 사용
   './weights/y5s_digit_best.pt',
   source='local' # 로컬 저장소 사용
   )
#    pretrained=True)

#%%
# Images

imgs = ['./res/d_01.jpg']  # batch of images

model.conf = 0.5  # NMS confidence threshold
model.iou = 0.2  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
model.max_det = 1000  # maximum number of detections per image
model.amp = False  # Automatic Mixed Precision (AMP) inference


# model.cpu() 
# Inference
results = model(imgs)

# Results
results.print()
results.save()  # or .show() 결과 이미지 저장

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)
# %% json results
results.pandas().xyxy[0].to_json(orient="records") 
# %%
