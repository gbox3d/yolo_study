import uuid
import numpy as np
import json
import os
from datetime import datetime

class LabelManager:
    """
    단일 프레임에 대한 라벨 목록을 관리합니다.
    각 라벨은 {'id': str, 'class_id': int, 'class_name': str, 'bbox_processed': [x,y,w,h]} 형태입니다.
    bbox_processed는 처리 해상도(예: 640x640) 기준입니다.
    """
    def __init__(self):
        self.labels = []

    def add_label(self, class_id, class_name, bbox_processed):
        """새 라벨을 추가합니다."""
        if not bbox_processed or len(bbox_processed) != 4:
            print("Error: Invalid bbox_processed for add_label.")
            return None
        new_id = uuid.uuid4().hex
        new_label = {
            'id': new_id,
            'class_id': class_id,
            'class_name': class_name,
            'bbox_processed': list(bbox_processed) # Ensure it's a list copy
        }
        self.labels.append(new_label)
        print(f"LabelManager: 라벨 추가됨 - ID {new_id}")
        return new_id

    def update_label(self, label_id, class_id, class_name, bbox_processed):
        """기존 라벨을 업데이트합니다."""
        if not bbox_processed or len(bbox_processed) != 4:
            print("Error: Invalid bbox_processed for update_label.")
            return False
        for i, label in enumerate(self.labels):
            if label.get('id') == label_id:
                self.labels[i]['class_id'] = class_id
                self.labels[i]['class_name'] = class_name
                self.labels[i]['bbox_processed'] = list(bbox_processed)
                print(f"LabelManager: 라벨 업데이트됨 - ID {label_id}")
                return True
        print(f"LabelManager: 업데이트할 라벨 ID {label_id} 찾지 못함.")
        return False

    def delete_label(self, label_id):
        """특정 ID의 라벨을 삭제합니다."""
        initial_len = len(self.labels)
        self.labels = [label for label in self.labels if label.get('id') != label_id]
        if len(self.labels) < initial_len:
            print(f"LabelManager: 라벨 삭제됨 - ID {label_id}")
            return True
        print(f"LabelManager: 삭제할 라벨 ID {label_id} 찾지 못함.")
        return False

    def get_labels(self):
        """현재 프레임의 모든 라벨 목록을 반환합니다."""
        return self.labels

    def get_label_by_id(self, label_id):
        """특정 ID의 라벨을 반환합니다."""
        for label in self.labels:
            if label.get('id') == label_id:
                return label
        return None

    def to_yolo_format_strings(self, processing_width, processing_height):
        """
        현재 관리 중인 모든 라벨을 YOLO 형식의 문자열 리스트로 변환합니다.
        <class_id> <x_center_rel> <y_center_rel> <width_rel> <height_rel>
        """
        yolo_lines = []
        if processing_width == 0 or processing_height == 0:
            print("Error: Processing dimensions are zero, cannot convert to YOLO format.")
            return []
            
        for label in self.labels:
            class_id = label['class_id']
            x_proc, y_proc, w_proc, h_proc = label['bbox_processed']

            x_center_rel = (x_proc + w_proc / 2) / processing_width
            y_center_rel = (y_proc + h_proc / 2) / processing_height
            width_rel = w_proc / processing_width
            height_rel = h_proc / processing_height
            
            # 값 범위 확인 및 클리핑 (0.0 ~ 1.0)
            x_center_rel = np.clip(x_center_rel, 0.0, 1.0)
            y_center_rel = np.clip(y_center_rel, 0.0, 1.0)
            width_rel = np.clip(width_rel, 0.0, 1.0)
            height_rel = np.clip(height_rel, 0.0, 1.0)

            yolo_lines.append(f"{class_id} {x_center_rel:.6f} {y_center_rel:.6f} {width_rel:.6f} {height_rel:.6f}")
        return yolo_lines

    def clear_labels(self):
        """모든 라벨을 초기화합니다."""
        self.labels = []
        print("LabelManager: 모든 라벨이 초기화되었습니다.")

    def save_to_file(self, filepath):
        """라벨 데이터를 JSON 파일로 저장합니다."""
        try:
            save_data = {
                'metadata': {
                    'version': '1.0',
                    'created_at': datetime.now().isoformat(),
                    'label_count': len(self.labels)
                },
                'labels': self.labels
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            print(f"LabelManager: 라벨 데이터 저장 완료 - {filepath}")
            return True, f"라벨 데이터가 성공적으로 저장되었습니다.\n파일: {filepath}"
            
        except Exception as e:
            error_msg = f"라벨 데이터 저장 중 오류 발생: {e}"
            print(f"LabelManager Error: {error_msg}")
            return False, error_msg

    def load_from_file(self, filepath):
        """JSON 파일에서 라벨 데이터를 로드합니다."""
        try:
            if not os.path.exists(filepath):
                return False, f"파일을 찾을 수 없습니다: {filepath}"
            
            with open(filepath, 'r', encoding='utf-8') as f:
                load_data = json.load(f)
            
            if 'labels' not in load_data:
                return False, "잘못된 라벨 파일 형식입니다."
            
            # 기존 라벨 초기화 후 로드
            self.labels = []
            loaded_labels = load_data['labels']
            
            for label_data in loaded_labels:
                # 필수 필드 검증
                required_fields = ['class_id', 'class_name', 'bbox_processed']
                if not all(field in label_data for field in required_fields):
                    print(f"LabelManager Warning: 불완전한 라벨 데이터 건너뜀 - {label_data}")
                    continue
                
                # ID가 없으면 새로 생성
                if 'id' not in label_data:
                    label_data['id'] = uuid.uuid4().hex
                
                # bbox_processed 검증
                bbox = label_data['bbox_processed']
                if not bbox or len(bbox) != 4:
                    print(f"LabelManager Warning: 잘못된 bbox 데이터 건너뜀 - {bbox}")
                    continue
                
                self.labels.append(label_data)
            
            metadata = load_data.get('metadata', {})
            loaded_count = len(self.labels)
            original_count = metadata.get('label_count', loaded_count)
            
            print(f"LabelManager: 라벨 데이터 로드 완료 - {filepath}")
            return True, f"라벨 데이터가 성공적으로 로드되었습니다.\n로드된 라벨: {loaded_count}개 (원본: {original_count}개)"
            
        except Exception as e:
            error_msg = f"라벨 데이터 로드 중 오류 발생: {e}"
            print(f"LabelManager Error: {error_msg}")
            return False, error_msg

    def export_to_yolo_files(self, output_dir, filename_prefix, processing_width, processing_height):
        """라벨 데이터를 YOLO 형식 텍스트 파일로 내보냅니다."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            yolo_strings = self.to_yolo_format_strings(processing_width, processing_height)
            txt_filepath = os.path.join(output_dir, f"{filename_prefix}.txt")
            
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                if yolo_strings:
                    f.write("\n".join(yolo_strings))
                else:
                    f.write("")  # 빈 파일 생성
            
            print(f"LabelManager: YOLO 파일 내보내기 완료 - {txt_filepath}")
            return True, txt_filepath, f"YOLO 형식으로 내보내기 완료\n파일: {txt_filepath}\n라벨 수: {len(yolo_strings)}개"
            
        except Exception as e:
            error_msg = f"YOLO 파일 내보내기 중 오류 발생: {e}"
            print(f"LabelManager Error: {error_msg}")
            return False, None, error_msg

    def import_from_yolo_file(self, yolo_filepath, class_names_dict, processing_width, processing_height):
        """YOLO 형식 텍스트 파일에서 라벨 데이터를 가져옵니다."""
        try:
            if not os.path.exists(yolo_filepath):
                return False, f"YOLO 파일을 찾을 수 없습니다: {yolo_filepath}"
            
            imported_labels = []
            with open(yolo_filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"LabelManager Warning: 잘못된 YOLO 형식 (라인 {line_num}): {line}")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        cx_rel, cy_rel, w_rel, h_rel = map(float, parts[1:])
                        
                        # 상대 좌표를 처리 해상도 절대 좌표로 변환
                        x_proc = int((cx_rel - w_rel / 2) * processing_width)
                        y_proc = int((cy_rel - h_rel / 2) * processing_height)
                        w_proc = int(w_rel * processing_width)
                        h_proc = int(h_rel * processing_height)
                        
                        # 경계 검사
                        x_proc = max(0, min(x_proc, processing_width - 1))
                        y_proc = max(0, min(y_proc, processing_height - 1))
                        w_proc = max(1, min(w_proc, processing_width - x_proc))
                        h_proc = max(1, min(h_proc, processing_height - y_proc))
                        
                        class_name = class_names_dict.get(class_id, f"Unknown_{class_id}")
                        
                        label_data = {
                            'id': uuid.uuid4().hex,
                            'class_id': class_id,
                            'class_name': class_name,
                            'bbox_processed': [x_proc, y_proc, w_proc, h_proc]
                        }
                        imported_labels.append(label_data)
                        
                    except ValueError as e:
                        print(f"LabelManager Warning: 숫자 변환 오류 (라인 {line_num}): {line} - {e}")
                        continue
            
            # 기존 라벨에 추가
            original_count = len(self.labels)
            self.labels.extend(imported_labels)
            
            print(f"LabelManager: YOLO 파일 가져오기 완료 - {yolo_filepath}")
            return True, f"YOLO 파일에서 가져오기 완료\n기존 라벨: {original_count}개\n추가된 라벨: {len(imported_labels)}개\n총 라벨: {len(self.labels)}개"
            
        except Exception as e:
            error_msg = f"YOLO 파일 가져오기 중 오류 발생: {e}"
            print(f"LabelManager Error: {error_msg}")
            return False, error_msg

    def get_statistics(self):
        """라벨 통계 정보를 반환합니다."""
        if not self.labels:
            return "라벨이 없습니다."
        
        # 클래스별 개수
        class_counts = {}
        total_area = 0
        
        for label in self.labels:
            class_name = label['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            bbox = label['bbox_processed']
            area = bbox[2] * bbox[3]  # width * height
            total_area += area
        
        stats = [f"총 라벨 수: {len(self.labels)}"]
        stats.append(f"평균 바운딩 박스 면적: {total_area / len(self.labels):.1f}px²")
        stats.append("\n클래스별 개수:")
        
        for class_name, count in sorted(class_counts.items()):
            percentage = (count / len(self.labels)) * 100
            stats.append(f"  {class_name}: {count}개 ({percentage:.1f}%)")
        
        return "\n".join(stats)