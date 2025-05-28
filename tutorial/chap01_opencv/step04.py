import cv2
import time

# OpenCV를 사용하여 카메라에서 프레임을 읽고, 특정 영역(ROI)을 잘라내어 저장하는 예제입니다.

# 1) 카메라 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라 열기 실패")
else:
    # 속성 조회
    print("Width:",  cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("FPS:",    cap.get(cv2.CAP_PROP_FPS))

time.sleep(1)  # 카메라 초기화 대기

# 3) 프레임 읽기
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("프레임을 읽어오지 못했습니다")

# 4) ROI 좌표 설정 (x, y: 좌상단 / w, h: 너비·높이)
x, y, w, h = 100, 50, 200, 150

# 5) 원본에 녹색 사각형 그리기
#    cv2.rectangle(img, (x1,y1), (x2,y2), color(B,G,R), thickness)
annotated = frame.copy()
cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 6) ROI 잘라내기
roi = frame[y:y + h, x:x + w]

# 7) 파일로 저장
cv2.imwrite("annotated_original.png", annotated)  # 녹색 사각형 표시된 원본
cv2.imwrite("cropped_roi.png", roi)               # 잘라낸 영역

# 8) (선택) 화면에 띄워보고 대기
cv2.imshow("Annotated Original", annotated)
cv2.imshow("Cropped ROI", roi)
cv2.waitKey(0) # 아무키나 누르면 종료
cv2.destroyAllWindows()
