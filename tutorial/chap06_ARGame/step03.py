import cv2
import pygame
import sys
from ultralytics import SAM
import numpy as np


width, height = 640, 480  # Pygame 창 크기 설정

# --- 0. 초기 설정 ---
# 1) SAM 모델 로드
MODEL_NAME = "sam2_s.pt" # 또는 사용 가능한 SAM 모델 (예: "sam_s.pt", "mobile_sam.pt")
try:
    model = SAM(MODEL_NAME)
    print(f"'{MODEL_NAME}' 모델을 성공적으로 로드했습니다.")
except Exception as e:
    print(f"'{MODEL_NAME}' 모델 로드 중 오류 발생: {e}")
    print(f"'{MODEL_NAME}' 파일이 현재 작업 디렉토리에 있는지, ultralytics.SAM이 해당 모델을 지원하는지,")
    print(f"또는 PyTorch 버전과 CUDA 가용성을 확인해주세요.")
    sys.exit()

# 2) 비디오 파일 열기
VIDEO_PATH = "tutorial/chap06_ARGame/by_drone.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"오류: 비디오 파일을 열 수 없습니다. 경로를 확인하세요: {VIDEO_PATH}")
    sys.exit()

video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps == 0:
    video_fps = 30  # 기본 FPS 설정
print(f"비디오 FPS: {video_fps}")

# 3) Pygame 초기화
pygame.init()
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
screen = pygame.display.set_mode((w, h))
pygame.display.set_caption("SAM 객체 선택 및 CSRT 추적")
clock = pygame.time.Clock()

font = pygame.font.SysFont(None,24)


# --- 상태 변수 ---
click_point = None
current_bbox = None    # 현재 프레임에 표시될 최종 바운딩 박스 (SAM 또는 추적기 결과)
sam_processing = False # SAM 모델이 현재 처리 중인지 여부
tracker = None         # OpenCV 추적기 객체
tracking_active = False # 추적기가 현재 활성화되어 객체를 추적 중인지 여부

# --- 도우미 함수 ---
def get_bbox_from_mask(mask_bool):
    if mask_bool is None or not np.any(mask_bool):
        return None
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_rect, h_rect = cv2.boundingRect(largest_contour)
        return (x, y, w_rect, h_rect)
    return None

#--- missile
missile_pos = []
missile_direction = []
missile_speed = 50 # 미사일 속도 (픽셀/초)
missile_is_active = False

delta_tick = 0

# --- 메인 루프 ---
running = True
while running:
    
    
    ret, frame_bgr = cap.read()
    if not ret:
        print("비디오 끝 또는 프레임 읽기 오류. 루프 종료.")
        
        break

    # 1. 사용자 클릭으로 SAM을 이용한 초기 객체 감지 및 추적기 초기화
    if click_point and not sam_processing and not tracking_active:
        sam_processing = True
        print(f"SAM 처리 시작 (클릭): 좌표 {click_point}")

        results = model(frame_bgr, points=[click_point], labels=[1])

        initial_bbox_from_sam = None
        if results and results[0].masks:
            mask_data = results[0].masks.data[0].cpu().numpy()
            mask_bool = mask_data.astype(bool)
            initial_bbox_from_sam = get_bbox_from_mask(mask_bool)
            
        if initial_bbox_from_sam:
            current_bbox = initial_bbox_from_sam # SAM 결과를 현재 바운딩 박스로 설정
            print(f"SAM (클릭) 바운딩 박스 생성 성공: {current_bbox}")
            
            try:
                # CSRT 추적기 생성 및 초기화
                # tracker.init()은 (x,y,w,h) 튜플을 받으며, get_bbox_from_mask가 이 형식으로 반환
                init_tracker_bbox = tuple(map(int, current_bbox)) # 정수형 좌표 보장
                tracker = cv2.TrackerCSRT_create() 
                tracker.init(frame_bgr, init_tracker_bbox) # 원본 프레임으로 초기화
                tracking_active = True
                print(f"CSRT 추적기 초기화 성공: {current_bbox}")
            except Exception as e:
                print(f"CSRT 추적기 초기화 실패: {e}")
                tracker = None
                tracking_active = False
                current_bbox = None # 추적기 초기화 실패 시 bbox도 제거
        else:
            print("SAM (클릭) 바운딩 박스 생성 실패.")
            current_bbox = None # SAM 실패 시 bbox 제거

        click_point = None # 클릭 포인트 처리 완료
        sam_processing = False

    # 2. 추적기가 활성화된 경우, 추적기 업데이트
    elif tracking_active and tracker:
        success, bbox_from_tracker = tracker.update(frame_bgr) # 원본 프레임으로 추적
        if success:
            current_bbox = tuple(map(int, bbox_from_tracker)) # 추적 결과를 현재 바운딩 박스로 업데이트
        else:
            print("CSRT 추적 실패. 객체를 놓쳤습니다.")
            tracking_active = False
            tracker = None
            current_bbox = None # 추적 실패 시 bbox 제거
            

    # 3. Pygame 화면 준비 및 그리기
    # OpenCV 프레임(BGR)을 Pygame Surface(RGB)로 변환
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pygame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    screen.blit(pygame_surface, (0, 0))
    
    # 현재 바운딩 박스가 있으면 Pygame Surface에 그리기
    if current_bbox:
        pygame.draw.rect(screen, (0, 255, 0), current_bbox, 2)
    
    if missile_is_active:
        # direction 벡터 계산
        if current_bbox:
            x, y, w, h = current_bbox
            target_center = [x + w // 2, y + h // 2]
            
            # 거리 계산
            distance_to_target = np.linalg.norm(np.array(target_center) - np.array(missile_pos))
            if distance_to_target < 10:  # 목표에 도달했을 때
                print("미사일이 목표에 도달했습니다.")
                missile_is_active = False
            
            # 미사일 방향 벡터 계산
            missile_direction = [target_center[0] - missile_pos[0], target_center[1] - missile_pos[1]]
            # 단위 벡터로
            norm = np.linalg.norm(missile_direction) # 벡터의 크기 계산
            if norm > 0:
                missile_direction = [missile_direction[0] / norm, missile_direction[1] / norm]
            else:
                missile_direction = [0, 0]
            
        
        # 미사일 위치 업데이트
        missile_pos[0] += missile_direction[0] * missile_speed * (delta_tick / 1000.0)  # delta_tick을 초 단위로 변환
        missile_pos[1] += missile_direction[1] * missile_speed * (delta_tick / 1000.0)  # delta_tick을 초 단위로 변환
        
        # red circle 그리기
        pygame.draw.circle(screen, (255, 0, 0), (int(missile_pos[0]), int(missile_pos[1])), 5)

    # 화면에 안내 메시지 및 바운딩 박스 정보 표시
    message_str = "left click: SAM object selection, right click: reset tracking"
    if sam_processing:
        message_str = "SAM process..."
    elif tracking_active and current_bbox:
        message_str = "CSRT tracking..."
    elif current_bbox: # SAM은 성공했으나 추적기가 비활성/실패한 경우
         message_str = "SAM bbox detected, but tracking inactive."

    text_surface = font.render(message_str, True, (0, 255, 0) if not sam_processing else (255,165,0))
    screen.blit(text_surface, (10, 10))
    
    pygame.display.flip()

    # FPS 조절
    delta_tick = clock.tick(video_fps)

    # 4. Pygame 이벤트 처리
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE:
                # 스페이스바를 누르면 미사일 발사
                if current_bbox and not missile_is_active:
                    missile_pos = [width // 2, height]  # 화면 중앙에서 발사
                    missile_direction = [0, 0]
                    missile_is_active = True
                    
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # 마우스 좌클릭
                # SAM 처리 중이 아니고, 현재 추적 중이지도 않을 때만 새 객체 선택 허용
                if not sam_processing and not tracking_active:
                    click_point = list(event.pos)
                    print(f"마우스 좌클릭: {click_point} (새 객체 선택 시도)")
                    current_bbox = None # 새 선택 시 이전 bbox 정보 초기화
                    if tracker: # 이전 추적기 객체가 있다면 None으로 초기화
                        tracker = None
                elif sam_processing:
                    print("현재 SAM 모델이 처리 중입니다. 잠시 기다려주세요.")
                elif tracking_active:
                    print("현재 다른 객체를 추적 중입니다. 우클릭으로 추적을 중지 후 시도하세요.")
            
            if event.button == 3: # 마우스 우클릭: 모든 감지 및 추적 상태 리셋
                print("마우스 우클릭: 모든 감지 및 추적 상태를 리셋합니다.")
                current_bbox = None
                click_point = None
                sam_processing = False
                tracker = None         # 추적기 객체 리셋
                tracking_active = False # 추적 상태 리셋
                

# --- 종료 처리 ---
cap.release()
pygame.quit()
sys.exit()