import cv2
import pygame
import sys

# 1) OpenCV 카메라 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    sys.exit()

# 2) Pygame 초기화
pygame.init()
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Step 1: 카메라 프레임 출력")

# ▶ Pygame 폰트 준비
font = pygame.font.SysFont(None, 24)

# ▶ 클릭 위치 리스트
click_points = []

# 3) 프레임 읽어 Pygame에 표시
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # BGR(OpenCV) → RGB(Pygame) 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Pygame Surface로 변환
    surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    screen.blit(surface, (0, 0)) # 화면에 프레임 표시
    
    # ▶ 1) 안내 문구
    text_surf = font.render("Press ESC to quit", True, (255,255,255))
    screen.blit(text_surf, (10,10))

    # ▶ 2) 중앙 사각형 (예: 200×150)
    rect_w, rect_h = 200, 150
    rect = pygame.Rect((width-rect_w)//2, (height-rect_h)//2, rect_w, rect_h)
    pygame.draw.rect(screen, (0,255,0), rect, 2)

    # ▶ 3) 마우스 위치 원 표시
    mx, my = pygame.mouse.get_pos()
    pygame.draw.circle(screen, (255,0,0), (mx, my), 5)
    
    # ▶ 저장된 클릭 위치에 십자 그리기
    for (cx, cy) in click_points:
        size = 10
        pygame.draw.line(screen, (0,0,255), (cx-size, cy), (cx+size, cy), 2)
        pygame.draw.line(screen, (0,0,255), (cx, cy-size), (cx, cy+size), 2)

    
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
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            # 왼쪽 클릭 시 좌표 저장
            click_points.append(e.pos)
