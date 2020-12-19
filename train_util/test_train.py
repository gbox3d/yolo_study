#%%
import cv2

from IPython.display import display
import PIL.Image as Image

import sys
sys.path.append('../libs')

from yolov5_myutil import createModel,predict

#%%
#config data 
import yaml
with open(f'test_conf.yaml', 'r') as f:
  config_data = yaml.load(f)

dataset_path = config_data['dataset_path'] # '/home/gbox3d/work/dataset/pistol/'
model_path = config_data['model_path'] #'/home/gbox3d/work/visionApp/yolov5/runs/train/gun_yolov5s_results/weights/best.pt'
test_index = config_data['test_index']

print(f'dataset : {dataset_path}')
print(f'model : {model_path}')

with open(f'{dataset_path}val.txt', 'r') as f:
  val_list = (f.read()).split('\n')

# print(val_list[3])
# %%

model,imgsz,names,device = createModel(model_path)

#%%
img0 = cv2.imread(val_list[test_index])  # BGR
pred,_img = predict(model,img0,device,imgsz,scaling=True)


print(img0.shape)
print(_img.shape)

# %%
result_img = img0.copy()
for i, det in enumerate(pred):
    # print(det)
    # Rescale boxes from img_size to im0 size
    # det[:, :4] = scale_coords(_img.shape[2:], det[:, :4], img0.shape).round()

    for *xyxy, conf, cls in reversed(det):
        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        print('%12s %.2f %.2f %.2f %.2f %.2f' % (names[int(cls)], conf,c1[0],c1[1],c2[0],c2[1]))
        # print(c1,c2)
        cv2.rectangle(result_img, c1, c2, (0,0,255), thickness=3, lineType=cv2.LINE_AA)
        
        
    # Print results
    for c in det[:, -1].unique():
        n = (det[:, -1] == c).sum()  # detections per class
        print('%g %ss, ' % (n, names[int(c)]))

display(Image.fromarray(cv2.cvtColor(result_img,cv2.COLOR_BGR2RGB)))


# %%
