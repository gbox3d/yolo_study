import cv2
import pygame
import sys
from ultralytics import YOLO, checks

checks()

# 1) OpenCV 카메라 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    sys.exit()

# 2) Pygame 초기화
pygame.init()
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
screen = pygame.display.set_mode((width, height))

model = YOLO("yolo11n.pt")
font = pygame.font.SysFont(None, 14)

# 3) 프레임 읽어 Pygame에 표시
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # ★ 추적 실행 : persist=True 로 이전 ID 유지
    results = model.track(frame,
                          conf=0.5,
                          persist=True,        # 같은 model 인스턴스면 자동 메모리 유지
                          verbose=False)[0]     # 첫 스트림만 사용

    # BGR(OpenCV) → RGB(Pygame) 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Pygame Surface로 변환
    surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1)) # OpenCV의 행렬을 Pygame Surface로 변환
    screen.blit(surface, (0, 0)) # 화면에 프레임 표시
    
    text_surf = font.render("Press ESC to quit", True, (255,255,255))
    screen.blit(text_surf, (10,10))
    
    
    # ★ 결과 렌더링
    if results.boxes:
        
        ids = results.boxes.id
        
        # ids 가 None 이면 [-1, -1, …] 로 채우기
        ids = [-1]*len(results.boxes.xyxy) if ids is None else ids.int().cpu().tolist()
        
        for box, tid in zip(results.boxes.xyxy.cpu(),
                            ids):  # track_id 목록
            x1, y1, x2, y2 = map(int, box)
            cls_id   = int(results.boxes.cls[0])
            cls_name = model.names[cls_id]
            conf     = results.boxes.conf[0].item() * 100

            # 바운딩 박스
            pygame.draw.rect(screen, (0, 255, 0), (x1, y1, x2-x1, y2-y1), 2)

            # 레이블 (클래스 + ID + 신뢰도)
            label = f"{cls_name} #{tid} {conf:.0f}%"
            text  = font.render(label, True, (0, 0, 0))
            tw, th = text.get_size()
            pygame.draw.rect(screen, (0, 255, 0), (x1, y1 - th - 2, tw + 4, th + 2))
            screen.blit(text, (x1 + 2, y1 - th))
    
    
    
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
