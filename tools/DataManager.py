import os
import yaml
import cv2
# from tkinter import messagebox # Removed Tkinter dependency

class DataManagerError(Exception):
    """Custom exception for DataManager errors."""
    pass

class DataManager:
    CONFIG_FILE = "config.yaml"

    def __init__(self):
        self.class_names = {0: "object"}
        self.class_ids = {"object": 0}
        self.save_base_path = "./labeled_data"
        self._config_loaded_successfully = False
        self.load_config()

    def is_config_loaded(self):
        return self._config_loaded_successfully

    def load_config(self):
        try:
            with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                if isinstance(config_data, dict):
                    if 'names' in config_data and isinstance(config_data['names'], dict):
                        self.class_names = {int(k): str(v) for k, v in config_data['names'].items()}
                        self.class_ids = {str(v): int(k) for k, v in config_data['names'].items()}
                        print(f"DataManager: 클래스 로드 성공 - {self.class_names}")
                    else:
                        print(f"DataManager Warning: {self.CONFIG_FILE}에서 'names'를 찾을 수 없거나 형식이 잘못되었습니다. 기본값을 사용합니다.")
                    
                    if 'path' in config_data and isinstance(config_data['path'], str):
                        self.save_base_path = config_data['path']
                        print(f"DataManager: 저장 경로 로드 성공 - {self.save_base_path}")
                    else:
                        print(f"DataManager Warning: {self.CONFIG_FILE}에서 'path'를 찾을 수 없거나 형식이 잘못되었습니다. 기본 경로 '{self.save_base_path}'를 사용합니다.")
                    self._config_loaded_successfully = True
                else:
                    # raise DataManagerError(f"{self.CONFIG_FILE} 파일의 최상위 형식이 딕셔너리가 아닙니다.")
                    print(f"DataManager Error: {self.CONFIG_FILE} 파일의 최상위 형식이 딕셔너리가 아닙니다. 기본 설정을 사용합니다.")
        except FileNotFoundError:
            print(f"DataManager Warning: {self.CONFIG_FILE} 파일을 찾을 수 없습니다. 기본 설정을 사용합니다.")
        except Exception as e:
            # raise DataManagerError(f"{self.CONFIG_FILE} 파일 로드 중 오류 발생: {e}")
            print(f"DataManager Error: {self.CONFIG_FILE} 파일 로드 중 오류 발생: {e}. 기본 설정을 사용합니다.")

    def get_class_names_dict(self):
        return self.class_names

    def get_class_ids_dict(self):
        return self.class_ids
        
    def get_save_base_path(self):
        return self.save_base_path

    def save_image_and_yolo_labels(self, original_frame, yolo_label_strings, prefix, frame_number_str_padded):
        images_dir = os.path.join(self.save_base_path, "images")
        labels_dir = os.path.join(self.save_base_path, "labels")
        
        try:
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
        except OSError as e:
            print(f"DataManager Error: 저장 폴더 생성에 실패했습니다 - {e}")
            return False, None, None, f"저장 폴더 생성 실패: {e}"

        base_filename = f"{prefix}_{frame_number_str_padded}"
        img_filepath = os.path.join(images_dir, f"{base_filename}.jpg")
        txt_filepath = os.path.join(labels_dir, f"{base_filename}.txt")

        img_saved = False
        try:
            cv2.imwrite(img_filepath, original_frame)
            print(f"DataManager: 이미지 저장 성공 - {img_filepath}")
            img_saved = True
        except Exception as e:
            print(f"DataManager Error: 이미지 저장 중 오류 발생 - {e}")
            return False, img_filepath, None, f"이미지 저장 오류: {e}"
        
        try:
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_label_strings))
            print(f"DataManager: 라벨 저장 성공 - {txt_filepath}")
            return True, img_filepath, txt_filepath, "저장 완료"
        except Exception as e:
            print(f"DataManager Error: 라벨 파일 저장 중 오류 발생 - {e}")
            # 이미지가 이미 저장되었을 수 있으므로, 이미지 경로도 반환
            return False, img_filepath if img_saved else None, txt_filepath, f"라벨 파일 저장 오류: {e}"


    def list_labeled_images(self):
        images_dir = os.path.join(self.save_base_path, "images")
        if not os.path.isdir(images_dir):
            print(f"DataManager Info: 이미지 디렉토리({images_dir})를 찾을 수 없습니다.")
            return []
        
        try:
            image_files = [f for f in os.listdir(images_dir) 
                           if os.path.isfile(os.path.join(images_dir, f)) and 
                              f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            return sorted(image_files)
        except Exception as e:
            print(f"DataManager Error: 저장된 이미지 목록 읽기 오류 - {e}")
            return []

    def load_yolo_labels_from_file(self, image_filename_with_ext):
        base_filename = os.path.splitext(image_filename_with_ext)[0]
        labels_dir = os.path.join(self.save_base_path, "labels")
        txt_filepath = os.path.join(labels_dir, f"{base_filename}.txt")

        labels = [] 
        if os.path.exists(txt_filepath):
            try:
                with open(txt_filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            try:
                                class_id = int(parts[0])
                                cx, cy, w, h = map(float, parts[1:])
                                labels.append({'class_id': class_id, 
                                               'class_name': self.class_names.get(class_id, f"ID:{class_id}"), # 클래스 이름 없으면 ID 표시
                                               'bbox_yolo': [cx, cy, w, h]})
                            except ValueError:
                                print(f"DataManager Warning: 라벨 파일 형식 오류 (숫자 변환 실패) - {txt_filepath}, line: {line.strip()}")
                        elif line.strip():
                             print(f"DataManager Warning: 라벨 파일 형식 오류 (항목 개수) - {txt_filepath}, line: {line.strip()}")
                # print(f"DataManager: 라벨 로드 성공 - {txt_filepath}, {len(labels)}개 라벨")
            except Exception as e:
                print(f"DataManager Error: 라벨 파일 읽기 오류 - {txt_filepath}, {e}")
        else:
            # print(f"DataManager Info: 라벨 파일을 찾을 수 없음 (정상일 수 있음) - {txt_filepath}")
            pass # 라벨 파일 없는 것은 오류가 아님 (제로 라벨링)
        return labels
        
    def get_image_path(self, image_filename_with_ext):
        images_dir = os.path.join(self.save_base_path, "images")
        return os.path.join(images_dir, image_filename_with_ext)

