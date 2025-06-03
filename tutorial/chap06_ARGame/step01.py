import cv2
import pygame
import sys
from ultralytics import SAM # ultralytics.SAM이 SAM2 모델도 로드할 수 있는지 확인 필요
import numpy as np

# --- 0. 초기 설정 ---
# 1) SAM 모델 로드
MODEL_NAME = "sam2_s.pt" # 사용자님께서 명시하신 모델명
try:
    model = SAM(MODEL_NAME)
    print(f"'{MODEL_NAME}' 모델을 성공적으로 로드했습니다.")
except Exception as e:
    print(f"'{MODEL_NAME}' 모델 로드 중 오류 발생: {e}")
    print(f"'{MODEL_NAME}' 파일이 현재 작업 디렉토리에 있는지, ultralytics.SAM이 해당 모델을 지원하는지 확인해주세요.")
    sys.exit()

# 2) 비디오 파일 열기
VIDEO_PATH = "tutorial/chap06_ARGame/by_drone.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"오류: 비디오 파일을 열 수 없습니다. 경로를 확인하세요: {VIDEO_PATH}")
    sys.exit()

# 비디오 FPS 정보 얻기
video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps == 0:
    video_fps = 30
print(f"비디오 FPS: {video_fps}")

# 3) Pygame 초기화
pygame.init()
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
screen = pygame.display.set_mode((w, h))
pygame.display.set_caption("SAM2 프롬프트 기반 객체 재탐지")
clock = pygame.time.Clock()

# Pygame 폰트 설정
try:
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 28)
except pygame.error as e:
    print(f"Pygame 폰트 로드 오류: {e}. 기본 시스템 폰트를 사용합니다.")
    font = pygame.font.SysFont(pygame.font.get_default_font(), 36)
    small_font = pygame.font.SysFont(pygame.font.get_default_font(), 28)

# --- 상태 변수 ---
click_point = None
current_bbox = None     # 현재 프레임에 표시될 최종 바운딩 박스
previous_bbox = None    # 이전 프레임에서 성공적으로 감지된 바운딩 박스 (다음 프레임의 프롬프트로 사용)
sam_processing = False

# --- 도우미 함수 ---
def get_bbox_from_mask(mask_bool):
    if mask_bool is None or not np.any(mask_bool):
        return None
    # OpenCV findContours는 (채널 없는) 흑백 이미지를 입력으로 받음
    # mask_bool은 True/False이므로 uint8로 변환 (True=255, False=0)
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_rect, h_rect = cv2.boundingRect(largest_contour)
        return (x, y, w_rect, h_rect)
    return None

# --- 메인 루프 ---
running = True
while running:
    ret, frame_bgr = cap.read()
    if not ret:
        print("비디오 끝 또는 프레임 읽기 오류. 루프 종료.")
        # 비디오 반복 재생 (선택 사항)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # click_point = None
        # current_bbox = None
        # previous_bbox = None
        # sam_processing = False
        # continue
        break

    display_frame_bgr = frame_bgr.copy()

    # 1. 사용자 클릭으로 초기 객체 감지
    if click_point and not sam_processing:
        sam_processing = True
        print(f"SAM 처리 시작 (클릭): 좌표 {click_point}")

        # SAM 모델에 프레임과 클릭 포인트 전달
        results = model(frame_bgr, points=[click_point], labels=[1])

        newly_detected_bbox = None
        if results and results[0].masks: # 결과와 마스크가 있는지 확인
            mask_data = results[0].masks.data[0].cpu().numpy()
            mask_bool = mask_data.astype(bool)
            newly_detected_bbox = get_bbox_from_mask(mask_bool)

        if newly_detected_bbox:
            current_bbox = newly_detected_bbox
            previous_bbox = current_bbox # 다음 프레임의 프롬프트로 사용하기 위해 저장
            print(f"SAM (클릭) 바운딩 박스 생성 성공: {current_bbox}")
        else:
            print("SAM (클릭) 바운딩 박스 생성 실패.")
            current_bbox = None
            previous_bbox = None # 실패 시 이전 박스도 없음 (재시도 방지)

        click_point = None
        sam_processing = False

    # 2. 이전 바운딩 박스를 프롬프트로 사용하여 객체 재탐지 (클릭이 없을 때)
    elif previous_bbox and not sam_processing:
        sam_processing = True

        # previous_bbox (x,y,w,h)를 SAM 프롬프트 형식 (x1,y1,x2,y2)로 변환
        x_prev, y_prev, w_prev, h_prev = previous_bbox
        prompt_bbox_for_sam = [x_prev, y_prev, x_prev + w_prev, y_prev + h_prev]

        print(f"SAM 처리 시작 (프롬프트): 이전 Bbox {prompt_bbox_for_sam}")
        # SAM 모델에 프레임과 바운딩 박스 프롬프트 전달
        results = model(frame_bgr, bboxes=[prompt_bbox_for_sam])

        newly_re_detected_bbox = None
        if results and results[0].masks: # 결과와 마스크가 있는지 확인
            # 여러 마스크가 반환될 수 있으므로, 필요시 추가 로직 구현 (예: IoU 기반 선택)
            # 여기서는 간단히 첫 번째 마스크를 사용합니다.
            mask_data = results[0].masks.data[0].cpu().numpy()
            mask_bool = mask_data.astype(bool)
            newly_re_detected_bbox = get_bbox_from_mask(mask_bool)

        if newly_re_detected_bbox:
            current_bbox = newly_re_detected_bbox
            previous_bbox = current_bbox # 다음 프레임의 프롬프트로 사용하기 위해 업데이트
            print(f"SAM (프롬프트) 바운딩 박스 갱신 성공: {current_bbox}")
        else:
            print("SAM (프롬프트) 바운딩 박스 갱신 실패.")
            current_bbox = None # 실패 시 현재 박스 표시 안 함
            previous_bbox = None # 실패 시 다음 프레임에서 프롬프트 사용 중지

        sam_processing = False

    # 3. 현재 바운딩 박스 그리기 (존재하는 경우)
    if current_bbox:
        p1 = (current_bbox[0], current_bbox[1])
        p2 = (current_bbox[0] + current_bbox[2], current_bbox[1] + current_bbox[3])
        cv2.rectangle(display_frame_bgr, p1, p2, (0, 255, 0), 2, 1)

    # 4. Pygame 화면 준비 및 표시
    frame_rgb = cv2.cvtColor(display_frame_bgr, cv2.COLOR_BGR2RGB)
    pygame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    screen.blit(pygame_surface, (0, 0))

    # 화면에 안내 메시지 표시
    message = "드론을 마우스 좌클릭 (우클릭: 박스/프롬프트 제거)"
    if sam_processing:
        message = "SAM 처리 중..."
    elif current_bbox:
        message = "SAM 바운딩 박스 표시됨 (프롬프트로 자동 갱신 시도)"

    text_surface = font.render(message, True, (255, 255, 255) if not sam_processing else (255,255,0) )
    screen.blit(text_surface, (10, 10))

    if current_bbox:
        bbox_text = f"BBox: ({current_bbox[0]},{current_bbox[1]}), W:{current_bbox[2]}, H:{current_bbox[3]}"
        bbox_surface = small_font.render(bbox_text, True, (200, 200, 255))
        screen.blit(bbox_surface, (10, h - 30))

    pygame.display.flip()

    # FPS 조절
    clock.tick(video_fps)

    # 5. Pygame 이벤트 처리
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # 마우스 좌클릭
                if not sam_processing:
                    click_point = list(event.pos)
                    print(f"마우스 좌클릭: {click_point}")
                    # 새 클릭이 들어오면, 이전 상태와 관계없이 새로 시작
                    current_bbox = None
                    previous_bbox = None
                elif sam_processing:
                    print("현재 SAM 모델이 처리 중입니다. 잠시 기다려주세요.")

            if event.button == 3: # 마우스 우클릭: 현재/이전 바운딩 박스 및 클릭 포인트 제거
                print("마우스 우클릭: 모든 감지 상태를 리셋합니다.")
                current_bbox = None
                previous_bbox = None
                click_point = None
                sam_processing = False

# --- 종료 처리 ---
cap.release()
pygame.quit()
sys.exit()
