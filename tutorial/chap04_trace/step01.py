import cv2
import pygame
import sys
from ultralytics import YOLO, checks

checks()

# 1) OpenCV 카메라 열기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 너비 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 높이 설정
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    sys.exit()

# 2) Pygame 초기화
pygame.init()
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
screen = pygame.display.set_mode((width, height))

model = YOLO("yolo11n.pt")
font = pygame.font.SysFont(None, 20)

# 3) 프레임 읽어 Pygame에 표시
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # ★ 추적 실행 : persist=True 로 이전 ID 유지
    results = model.track(frame,
                          conf=0.5,
                          stream= True,  # 스트림 모드 비활성화
                          persist=True,        # 같은 model 인스턴스면 자동 메모리 유지
                          verbose=False)

    # BGR(OpenCV) → RGB(Pygame) 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Pygame Surface로 변환
    surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1)) # OpenCV의 행렬을 Pygame Surface로 변환
    screen.blit(surface, (0, 0)) # 화면에 프레임 표시
    
    # ★ 결과 렌더링
    for result in results:  # results 는 list
        if not result.boxes:
            continue
        
        for box in result.boxes:  # box 는 Box 객체
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            id = int(box.id) if box.id is not None else -1
            conf = box.conf[0] if box.conf is not None else 0.0
            label = f"ID: {id}, Conf: {conf:.2f}"
            pygame.draw.rect(screen, (0, 255, 0), (x1, y1, x2 - x1, y2 - y1), 2)
            text_surf = font.render(label, True, (255, 0, 0))
            screen.blit(text_surf, (x1, y1 - 20))
            
    # fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_text = font.render(f"FPS: {fps:.2f}", True, (0, 0,255))
    screen.blit(fps_text, (10, 30))
    
    
    
    pygame.display.flip() # 

    # Quit 이벤트 처리
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE: # ESC 키로 종료
                cap.release()
                pygame.quit()
                sys.exit()
                
    clock = pygame.time.Clock()
    clock.tick(30)  # FPS 제한 (30 FPS)
