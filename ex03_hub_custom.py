# referrence : https://github.com/ultralytics/yolov5/issues/36
#%% 
import torch
import json

#%%
# custom Model loading 
model = torch.hub.load(
   '../yolov5', # 저장소 위치
   'custom' , # 커스텀 웨이트 파일 사용
   './weights/y5s_digit_best.pt',
   source='local' # 로컬 저장소 사용
   )

#%%
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
# results.print()
# results.save()  # or .show() 결과 이미지 저장

print(results.xyxy[0])  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)

# %% use json results
# results.pandas().xyxy[0].to_json(orient="records") 
_json = json.loads(results.pandas().xyxy[0].to_json(orient="records"))

for _det in _json:
    print(_det['xmin'], _det['ymin'], _det['xmax'], _det['ymax'],
          _det['confidence'], _det['class'], _det['name'])

# %% use numpy
dfResult = results.pandas().xyxy[0]
for det in dfResult.to_numpy():
    x1,y1,x2,y2,confidence, class_id,name = det[:7] 
    print(int(x1), int(y1), int(x2), int(y2), float(confidence), int(class_id),
          name)

# %% use tensor
for i, pred in enumerate(results.xyxy[0]):
   x1,y1,x2,y2 = pred[:4]
   confidence = pred[4]
   class_id = pred[5]
   print(int(x1), int(y1), int(x2), int(y2), float(confidence), int(class_id))
   
# %%
