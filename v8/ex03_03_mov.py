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
cap = cv.VideoCapture('../1.mp4')
total_framecount = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) # 전체 프레임 구하기 

print(f'total frame count {total_framecount}')

frame_index = 20
#%%
if cap.isOpened() :
    
    while True:
        cap.set(cv.CAP_PROP_POS_FRAMES,frame_index) #프레임 선택
        ret, frame = cap.read()
        
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=0.5,verbose=False)  # predict on an image
        
        org_frame = frame.copy()
        
        for det in results:
            for kpts in det.keypoints.data:
                # Each keypoint is a 2D tensor
                for kpt in kpts:
                    x, y, _ = map(int, kpt)
                    cv.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    
        cv.putText(frame, f"frame :{frame_index} / {total_framecount}", (10, 50), cv.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 255), 1, cv.LINE_AA) 
        
        cv.imshow('frame',frame)
        cv.imshow('org_frame',org_frame)
        
        
        
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
     
        
    