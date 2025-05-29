import cv2
import pygame
import sys
from ultralytics import SAM
import numpy as np

# 1) 모델 로드
model = SAM("sam2_s.pt")

# 2) 카메라 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    sys.exit()

# 3) Pygame 초기화
pygame.init()
w, h = int(cap.get(3)), int(cap.get(4))
screen = pygame.display.set_mode((w, h))
clock = pygame.time.Clock()

# 상태 변수
click_point = None
mask_bool   = None  # 마스크 불리언 배열

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if click_point:
        results    = model(frame, points=[click_point], labels=[1])
        mask_bool  = results[0].masks.data[0].cpu().numpy().astype(bool)
        click_point = None

    # 합성
    if mask_bool is not None:
        # 컬러 마스크 & 투명 합성
        color_mask = np.zeros_like(frame, dtype=np.uint8)
        color_mask[mask_bool] = (0, 255, 0)
        disp = cv2.addWeighted(frame, 0.2, color_mask, 0.5, 0) # 투명도 조절
    else:
        disp = frame

    # Pygame에 출력
    surf = pygame.surfarray.make_surface(
        cv2.cvtColor(disp, cv2.COLOR_BGR2RGB).swapaxes(0,1)
    )
    screen.blit(surf, (0,0))
    pygame.display.flip()

    #이벤트 처리 (루프 최상단에 두는 게 권장)
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            cap.release(); pygame.quit(); sys.exit()
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            cap.release(); pygame.quit(); sys.exit()
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            click_point = list(e.pos)
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 3: # 우클릭 clear
            click_point = None
            mask_bool   = None
