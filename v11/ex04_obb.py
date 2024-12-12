#%%
import cv2 as cv
import numpy as np
from ultralytics import YOLO

from IPython.display import display
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image


# Load a model
model = YOLO("yolo11n-obb.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

#%%
# Predict with the model
im = cv.imread('../boats.jpg')
results = model(im)  # predict on an image
# %%
result_img = im.copy()
# 결과 시각화
for result in results:
    if result.obb is not None:
        for obb in result.obb:
            # OBB 좌표 추출
            xywhr = obb.xywhr.cpu().numpy().flatten()
            x_center, y_center, width, height, rotation = xywhr

            # 회전 각도를 라디안에서 도 단위로 변환
            angle = np.degrees(rotation)

            # RotatedRect 생성
            rect = ((x_center, y_center), (width, height), angle)

            # RotatedRect의 네 꼭지점 좌표 계산
            box = cv.boxPoints(rect)
            box = np.int0(box)

            # # 화살표 끝점 계산 (길이를 비율로 설정)
            # arrow_length = max(width, height) / 2
            # arrow_end_x = x_center + arrow_length
            # arrow_end_y = y_center - arrow_length
            
            # # 45 도 회전된 화살표 끝점 계산
            # arrow_end_x, arrow_end_y = cv.transform(np.array([[[arrow_end_x, arrow_end_y]]]), cv.getRotationMatrix2D((x_center, y_center), angle, 1))[0][0]
            

            # 화살표 색상 설정
            if obb.cls == 1:
                color = (0, 255, 0)  # 초록색
            else:
                color = (0, 255, 255)  # 노란색

            # 이미지에 회전된 바운딩 박스 그리기
            cv.drawContours(result_img, [box], 0, color, 2)

            # 이미지에 화살표 그리기
            # cv.arrowedLine(
            #     result_img,
            #     (int(x_center), int(y_center)),
            #     (int(arrow_end_x), int(arrow_end_y)),
            #     (0,0,255),  # 빨간색
            #     2,
            #     tipLength=0.3  # 화살표 머리 크기
            # )
            
display(Image.fromarray(cv.cvtColor(result_img, cv.COLOR_BGR2RGB)))
# %%
