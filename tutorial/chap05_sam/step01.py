import cv2
import pygame
import sys

from ultralytics import SAM
import numpy as np

# model load
model = SAM("sam2_s.pt")

width, height = 640, 480

# pygame init
pygame.init()
screen_surface = pygame.display.set_mode((width, height))

# OpenCV init
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

font = pygame.font.SysFont(None,24)
Clock = pygame.time.Clock()

click_point = None
mask_bool = None

bLoop = True
while bLoop:
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    screen_surface.blit(frame_surface, (0, 0))
    
    if click_point is not None:
        # SAM 모델은 BGR 프레임을 그대로 사용할 수 있습니다.
        results = model(frame, points=[click_point], labels=[1]) # labels=[1]은 해당 점이 객체 내부임을 의미
        
        # 결과에서 마스크 데이터 추출 (결과가 있고, 마스크 데이터가 있는지 확인)
        if results and results[0].masks is not None and len(results[0].masks.data) > 0:
            mask_bool = results[0].masks.data[0].cpu().numpy().astype(bool) # (H, W) 형태의 boolean 배열
        else:
            print("Warning: SAM did not return a mask for the point.")
            mask_bool = None # 마스크를 찾지 못하면 None으로 설정
        click_point = None # 다음 클릭을 위해 포인트 초기화

    if mask_bool is not None: # mask_bool은 (height, width) 형태
        # Pygame 서피스를 per-pixel alpha를 지원하도록 생성 (너비, 높이 순서)
        mask_pygame_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        mask_pygame_surface.fill((0, 0, 0, 0))  # 완전히 투명하게 초기화

        # NumPy 마스크 (H, W)를 Pygame surfarray 인덱싱 (W, H)에 맞게 전치
        mask_bool_swapped = mask_bool.T  # (width, height) 형태로 변경

        # 서피스 픽셀 배열에 접근 (이 배열들을 수정하면 서피스가 변경됨)
        # pixels_rgb의 형태는 (width, height, 3)
        # pixels_alpha의 형태는 (width, height)
        pixels_rgb = pygame.surfarray.pixels3d(mask_pygame_surface)
        pixels_alpha = pygame.surfarray.pixels_alpha(mask_pygame_surface)
        
        # 마스크 영역에 녹색 적용
        pixels_rgb[mask_bool_swapped] = [0, 255, 0]  # R, G, B 값 설정
        
        # 마스크 영역에 알파값 적용
        alpha_value = 128  # 반투명 (0-255 사이)
        pixels_alpha[mask_bool_swapped] = alpha_value
        # 마스크가 아닌 영역은 fill((0,0,0,0))에 의해 이미 알파값이 0임

        # 서피스 배열에 대한 참조를 삭제하여 락 해제 (중요)
        del pixels_rgb
        del pixels_alpha
        
        screen_surface.blit(mask_pygame_surface, (0, 0))

    #fps
    fps = Clock.get_fps()
    fps_text = font.render(f"FPS: {fps:.2f}", True, (0, 0, 255))
    screen_surface.blit(fps_text, (10, 10))
        
    
    pygame.display.flip()
    Clock.tick(30)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            bLoop = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                bLoop = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            click_point = event.pos
            print(f"Click at: {click_point}")
        
                
pygame.quit()
cap.release()

print("exit successfully")
