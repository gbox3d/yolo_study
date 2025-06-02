import pygame
import sys
import cv2

# 1) Pygame 초기화
pygame.init()

# 2) 화면 생성 (640×480, 더블 버퍼링)
screen = pygame.display.set_mode(
    (640, 480),
    pygame.DOUBLEBUF
)
pygame.display.set_caption("Step 01: 초기화와 Surface")

clock = pygame.time.Clock() # FPS 조절용
font = pygame.font.SysFont(None, 24) # 기본 폰트

# OpenCV 카메라 열기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 너비 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 높이 설정


if not cap.isOpened():
    print("카메라 열기 실패"); sys.exit()
    
    

# 클릭 위치를 저장할 리스트
click_points = []

while True: 
   
    # 1) 화면 클리어 → 뒷버퍼 준비
    
    ret, frame = cap.read()
    if not ret: break
    # # OpenCV BGR → Pygame RGB 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # NumPy → Pygame Surface
    surf = pygame.surfarray.make_surface(
        frame_rgb.swapaxes(0,1)
    )
    
    screen.blit(surf, (0,0)) # 클리어 대신 카메라 영상 으로 채우기
    

    # 2) 예시 도형 그리기
    
    # 중앙에 사각형 그리기
    pygame.draw.rect(screen, (0,255,0),
                     pygame.Rect(220,160,200,150), 2)
    
    # 마우스 위치에 원 그리기
    mx, my = pygame.mouse.get_pos()
    pygame.draw.circle(screen, (255,0,0), (mx, my), 5)

    # 3) 안내 문구 표시
    screen.blit(
      font.render("Press ESC to quit", True, (255,255,255)),
      (10,10)
    )
    
    fps = clock.get_fps()  # 실시간 FPS 계산
    fps_surf = font.render(f"FPS: {fps:.2f}", True, (255, 255, 0))
    screen.blit(fps_surf, (640 - fps_surf.get_width() - 10, 10))
    
    # 4) 저장된 클릭 위치에 십자선 그리기 ← 추가된 부분
    for cx, cy in click_points:
        size = 10
        # 수평선
        pygame.draw.line(screen, (0, 0, 255),
                         (cx - size, cy),
                         (cx + size, cy), 2)
        # 수직선
        pygame.draw.line(screen, (0, 0, 255),
                         (cx, cy - size),
                         (cx, cy + size), 2)

    # 5) 화면 교체
    pygame.display.flip()
    clock.tick(30)   # 초당 30FPS 고정
    
     # 이벤트 처리
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (
           e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            pygame.quit(); sys.exit()
        # 마우스 왼쪽 클릭 시 좌표 저장
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            click_points.append(e.pos)
