#%%
import cv2
import numpy as np

from IPython.display import display
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image

from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results

def result_to_json(result: Results, tracker=None):
    """
    Convert result from ultralytics YOLOv8 prediction to json format
    Parameters:
        result: Results from ultralytics YOLOv8 prediction
        tracker: DeepSort tracker
    Returns:
        result_list_json: detection result in json format
    """
    len_results = len(result.boxes)
    result_list_json = [
        {
            'class_id': int(result.boxes.cls[idx]),
            'class': result.names[int(result.boxes.cls[idx])],
            'confidence': float(result.boxes.conf[idx]),
            'bbox': {
                'x_min': int(result.boxes.boxes[idx][0]),
                'y_min': int(result.boxes.boxes[idx][1]),
                'x_max': int(result.boxes.boxes[idx][2]),
                'y_max': int(result.boxes.boxes[idx][3]),
            },
        } for idx in range(len_results)
    ]
    if result.masks is not None:
        for idx in range(len_results):
            result_list_json[idx]['mask'] = cv2.resize(result.masks.data[idx].cpu().numpy(), (result.orig_shape[1], result.orig_shape[0])).tolist()
            result_list_json[idx]['segments'] = result.masks.segments[idx].tolist()
    if tracker is not None:
        bbs = [
            (
                [
                    result_list_json[idx]['bbox']['x_min'],
                    result_list_json[idx]['bbox']['y_min'],
                    result_list_json[idx]['bbox']['x_max'] - result_list_json[idx]['bbox']['x_min'],
                    result_list_json[idx]['bbox']['y_max'] - result_list_json[idx]['bbox']['y_min']
                ],
                result_list_json[idx]['confidence'],
                result_list_json[idx]['class'],
            ) for idx in range(len_results)
        ]
        tracks = tracker.update_tracks(bbs, frame=result.orig_img)
        for idx in range(len(result_list_json)):
            track_idx = next((i for i, track in enumerate(tracks) if track.det_conf is not None and np.isclose(track.det_conf, result_list_json[idx]['confidence'])), -1)
            if track_idx != -1:
                result_list_json[idx]['object_id'] = int(tracks[track_idx].track_id)
    return result_list_json

#%%
img = cv2.imread("bus.jpg")
model = YOLO("yolov8n-seg.pt")  
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
    
    cv2.rectangle(_img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
    
    print(f'class : {int(box.cls)}')
    print(f'conf : {box.conf.item()}')
    
display(Image.fromarray(cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)))


# %%
print( result.speed)

# %%
print(result.masks.segments)
print(result.masks.data)

# %% mask 정보
for _mask_data in result.masks.data:
    mask_img = _mask_data.cpu().numpy()
    mask_img = np.where(mask_img > 0.5, 255, 0).astype('uint8')  # 0.5 이상인 값을 255로 바꾸고, uint8로 변경
    display(Image.fromarray(cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)))
    
#%% segment 정보
_seg_img = result.orig_img.copy()
img_h, img_w, _ = _seg_img.shape

for _segment in result.masks.segments:
    # 좌표값을 이미지 크기에 맞게 정수형으로 변환
    np_cnt = (_segment * [img_w,img_h]).astype(np.int32)    
    cv2.polylines(_seg_img, [np_cnt], True, (0, 255, 0), 2)

display(Image.fromarray(cv2.cvtColor(_seg_img, cv2.COLOR_BGR2RGB)))
    
# %%
