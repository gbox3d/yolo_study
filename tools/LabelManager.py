import uuid
import numpy as np

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
        self.labels = []
        print("LabelManager: 모든 라벨이 초기화되었습니다.")
