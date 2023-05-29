#%%
import cv2 as cv

# import ultralytics
from ultralytics import YOLO,checks

from IPython.display import display
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image


checks()
#%%
# Load a model
model = YOLO('yolov8n-pose.pt')  # load an official model
#%% load mp4
cap = cv.VideoCapture('../dive_480p.mp4')
total_framecount = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) # 전체 프레임 구하기 

print(f'total frame count {total_framecount}')

frame_index = 20
#%%
if cap.isOpened() :
    
    while True:
        cap.set(cv.CAP_PROP_POS_FRAMES,frame_index) #프레임 선택
        ret, frame = cap.read()
        
        cv.imshow('frame',frame)
        # img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # display( Image.fromarray(img_rgb) )
        
        _key = cv.waitKey(20)
        if _key & 0xFF == ord('a'):
            frame_index += 1
        elif _key & 0xFF == ord('s'):
            frame_index -= 1
        elif _key & 0xFF == ord('q'):
            break

#%%
cap.release()   
exit()

#%%
cap.set(cv.CAP_PROP_POS_FRAMES,frame_index) #프레임 선택
ret, frame = cap.read()
display( Image.fromarray(frame) )
#%%
     
        
    