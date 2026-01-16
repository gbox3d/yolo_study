# setup_configs.py
import sam2
import os
import shutil

# 1. 설치된 sam2 라이브러리 위치 찾기
sam2_path = os.path.dirname(sam2.__file__)
config_src = os.path.join(sam2_path, "configs")

# 2. 내 프로젝트 폴더로 복사하기
config_dst = "./configs"

if os.path.exists(config_dst):
    print(f"'{config_dst}' 폴더가 이미 있습니다. 기존 폴더를 사용합니다.")
else:
    try:
        shutil.copytree(config_src, config_dst)
        print(f"성공! 설정 파일들을 '{config_dst}'로 복사했습니다.")
    except Exception as e:
        print(f"복사 실패: {e}")
        print("라이브러리가 제대로 설치되지 않은 것 같습니다.")