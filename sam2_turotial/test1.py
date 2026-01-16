import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Hydra & OmegaConf 관련 (설정 파일 로딩용)
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sam2.sam2_image_predictor import SAM2ImagePredictor

def build_sam2_local(config_dir, config_name, ckpt_path, device):
    """
    라이브러리 내부 build_sam2 대신, 로컬 config 폴더를 강제로 참조하는 함수
    """
    # 1. 기존 Hydra 설정 초기화 (충돌 방지)
    GlobalHydra.instance().clear()
    
    # 2. 절대 경로로 변환 (Hydra는 절대 경로를 선호합니다)
    abs_config_dir = os.path.abspath(config_dir)
    
    # 3. 로컬 폴더에서 설정 로드
    with initialize_config_dir(config_dir=abs_config_dir, version_base=None):
        cfg = compose(config_name=config_name)
        
    # 4. 모델 생성 (instantiate)
    # YAML 설정대로 파이썬 객체를 생성합니다.
    model = instantiate(cfg.model, _recursive_=True)
    
    # 5. 체크포인트 가중치 로드
    if ckpt_path:
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            # SAM2 체크포인트는 보통 "model" 키 안에 가중치가 들어있습니다.
            if "model" in state_dict:
                model.load_state_dict(state_dict["model"])
            else:
                model.load_state_dict(state_dict)
        else:
            print(f"경고: 체크포인트 파일({ckpt_path})이 없습니다. 랜덤 가중치로 시작합니다.")
            
    model.to(device)
    model.eval()
    return model

# =========================================================
# 실행 설정
# =========================================================

# 1. 파일 및 경로 설정
# 주의: setup_configs.py로 생성된 "configs" 폴더가 현재 위치에 있어야 합니다.
# local_config_dir = "./configs"         
local_config_dir = "./configs/sam2"  # <-- '/sam2'를 뒤에 붙여주세요
config_filename = "sam2_hiera_l.yaml"  # 파일명 (경로 제외)
checkpoint_file = "./sam2_hiera_large.pt"
image_file = "sample1.jpg"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"디바이스: {device}")

# 2. 모델 로드 (커스텀 함수 사용)
try:
    print("모델 로딩 중...")
    sam2_model = build_sam2_local(local_config_dir, config_filename, checkpoint_file, device)
    predictor = SAM2ImagePredictor(sam2_model)
    print("성공: 모델이 로드되었습니다!")
except Exception as e:
    print(f"\n[치명적 오류 발생]")
    print(f"1. './configs' 폴더가 있는지 확인하세요. (setup_configs.py 실행 필요)")
    print(f"2. 에러 내용: {e}")
    exit()

# 3. 이미지 테스트
if os.path.exists(image_file):
    image = Image.open(image_file).convert("RGB")
    image_np = np.array(image)
    
    predictor.set_image(image_np)
    
    # 임의 좌표 추론 (530, 130)
    input_point = np.array([[530, 130]])
    input_label = np.array([1])
    
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    
    best_idx = np.argmax(scores)
    print(f"추론 완료. 최고 점수: {scores[best_idx]:.4f}")
    
    # 시각화
    plt.figure(figsize=(10, 5))
    plt.imshow(image_np)
    plt.title(f"SAM2 Result: {scores[best_idx]:.2f}")
    plt.axis('off')
    
    # 마스크 오버레이 (간단 버전)
    mask = masks[best_idx]
    mask = mask.astype(bool)
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask, :] = [0, 1, 0, 0.5] # 초록색 반투명
    plt.imshow(overlay)
    plt.scatter(530, 130, c='red', marker='*', s=200) # 클릭 포인트
    plt.show()

else:
    print(f"이미지 파일({image_file})이 없어 추론 테스트를 건너뜁니다.")