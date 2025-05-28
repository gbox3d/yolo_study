#%% 시ㄹ시간 출력
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라 열기 실패")
else:
    # 속성 조회
    print("Width:",  cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("FPS:",    cap.get(cv2.CAP_PROP_FPS))

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
