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

# 3) 프레임 읽어 Pygame에 표시
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # BGR(OpenCV) → RGB(Pygame) 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Pygame Surface로 변환
    surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1)) # OpenCV의 행렬을 Pygame Surface로 변환
    screen.blit(surface, (0, 0)) # 화면에 프레임 표시
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
