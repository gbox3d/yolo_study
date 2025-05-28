#%%
import cv2
import time
import numpy as np
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라 열기 실패")
else:
    # 속성 조회
    print("Width:",  cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("FPS:",    cap.get(cv2.CAP_PROP_FPS))

time.sleep(1)  # 카메라 초기화 대기
    
#%%
ret, frame = cap.read()
if ret:
    cv2.imwrite("capture.png", frame)
cap.release()

# %%
