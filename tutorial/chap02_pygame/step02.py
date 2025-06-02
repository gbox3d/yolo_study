import pygame
import sys
import cv2


width, height = 640, 480

# 1) Pygame 초기화
pygame.init()

# 2) 화면 생성 (640×480, 더블 버퍼링)
screen = pygame.display.set_mode(
    (width, height),
    pygame.DOUBLEBUF
)
pygame.display.set_caption("Step 01: 초기화와 Surface")

clock = pygame.time.Clock() # FPS 조절용
font = pygame.font.SysFont(None, 24) # 기본 폰트

# OpenCV 카메라 열기
cap = cv2.VideoCapture(0)

# 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# 카메라가 열렸는지 확인
if not cap.isOpened():
    print("카메라 열기 실패"); sys.exit()

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

    # 4) 화면 교체
    pygame.display.flip()
    clock.tick(30)   # 초당 30FPS 고정
    
     # 이벤트 처리
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (
           e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            pygame.quit(); sys.exit()
