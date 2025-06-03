import cv2
import numpy as np
from PIL import Image, ImageTk

class VideoFrameRenderer:
    """
    비디오 프레임 렌더링 전담 클래스
    라벨, 마스크, 바운딩 박스 등의 시각적 요소를 그리는 역할
    """
    
    def __init__(self, processing_width=640, processing_height=640):
        self.processing_width = processing_width
        self.processing_height = processing_height
        
        # 색상 상수
        self.HANDLE_SIZE = 8
        self.HANDLE_COLOR = (255, 165, 0)  # BGR
        self.ACTIVE_BBOX_COLOR = (0, 255, 255)  # BGR  
        self.SAVED_BBOX_COLOR = (255, 0, 255)  # BGR
        self.POLYGON_COLOR = (0, 0, 255)  # BGR
        self.MASK_COLOR = (0, 255, 0)  # BGR

    def render_frame_with_labels(self, base_frame, frame_labels, active_elements=None, selected_class_name=""):
        """
        프레임에 라벨들을 렌더링합니다.
        
        Args:
            base_frame: 기본 프레임 (처리 해상도)
            frame_labels: 저장된 라벨 목록
            active_elements: 현재 작업 중인 요소들 dict
                - bbox: 활성 바운딩 박스
                - mask: 활성 마스크  
                - polygons: 활성 폴리곤들
                - selected_id: 선택된 라벨 ID
            selected_class_name: 선택된 클래스 이름
            
        Returns:
            렌더링된 프레임 (numpy array)
        """
        display_frame = base_frame.copy()
        
        # 저장된 라벨들 그리기
        if frame_labels:
            self._draw_saved_labels(display_frame, frame_labels, 
                                  skip_id=active_elements.get('selected_id') if active_elements else None)
        
        # 활성 요소들 그리기
        if active_elements:
            self._draw_active_elements(display_frame, active_elements, selected_class_name)
        
        return display_frame

    def _draw_saved_labels(self, frame, labels, skip_id=None):
        """저장된 라벨들을 그립니다."""
        for label in labels:
            # 현재 편집 중인 라벨은 건너뛰기
            if skip_id and label.get('id') == skip_id:
                continue
            
            x, y, w, h = label['bbox_processed']
            class_name = label['class_name']
            
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.SAVED_BBOX_COLOR, 2)
            
            # 클래스 이름 표시
            self._draw_text(frame, class_name, (x, y - 5), self.SAVED_BBOX_COLOR)

    def _draw_active_elements(self, frame, active_elements, selected_class_name):
        """현재 작업 중인 요소들을 그립니다."""
        active_bbox = active_elements.get('bbox')
        active_mask = active_elements.get('mask')
        active_polygons = active_elements.get('polygons', [])
        
        if active_bbox is not None:
            self._draw_active_bbox_with_handles(frame, active_bbox, selected_class_name)
            
            # 폴리곤도 함께 그리기
            if active_polygons:
                self._draw_polygons(frame, active_polygons)
                
        elif active_mask is not None:
            self._draw_mask_overlay(frame, active_mask)

    def _draw_active_bbox_with_handles(self, frame, bbox, class_name):
        """활성 바운딩 박스와 핸들을 그립니다."""
        x, y, w, h = bbox
        
        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.ACTIVE_BBOX_COLOR, 2)
        
        # 클래스 이름 표시
        if class_name:
            self._draw_text(frame, class_name, (x, y - 5), self.ACTIVE_BBOX_COLOR)
        
        # 핸들 그리기
        self._draw_bbox_handles(frame, bbox)

    def _draw_bbox_handles(self, frame, bbox):
        """바운딩 박스 모서리에 핸들을 그립니다."""
        x, y, w, h = bbox
        
        handle_positions = {
            'tl': (x, y),                    # top-left
            'tr': (x + w, y),               # top-right  
            'bl': (x, y + h),               # bottom-left
            'br': (x + w, y + h)            # bottom-right
        }
        
        for pos_name, (hx, hy) in handle_positions.items():
            cv2.rectangle(
                frame,
                (hx - self.HANDLE_SIZE // 2, hy - self.HANDLE_SIZE // 2),
                (hx + self.HANDLE_SIZE // 2, hy + self.HANDLE_SIZE // 2),
                self.HANDLE_COLOR,
                -1  # 채우기
            )

    def _draw_polygons(self, frame, polygons):
        """폴리곤들을 그립니다."""
        for polygon in polygons:
            if polygon is not None and len(polygon) > 0:
                cv2.drawContours(frame, [polygon.astype(np.int32)], -1, self.POLYGON_COLOR, 2)

    def _draw_mask_overlay(self, frame, mask):
        """마스크 오버레이를 그립니다."""
        if mask is not None:
            color_mask_overlay = np.zeros_like(frame, dtype=np.uint8)
            color_mask_overlay[mask == 1] = self.MASK_COLOR
            cv2.addWeighted(color_mask_overlay, 0.3, frame, 0.7, 0, dst=frame)

    def _draw_text(self, frame, text, position, color):
        """텍스트를 그립니다."""
        x, y = position
        # y가 너무 위에 있으면 아래로 이동
        if y < 15:
            y = 15
        
        cv2.putText(
            frame, 
            text, 
            (x, y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            color, 
            1,
            cv2.LINE_AA
        )

    def frame_to_photoimage(self, cv_frame, target_width, target_height):
        """
        OpenCV 프레임을 PhotoImage로 변환합니다.
        
        Args:
            cv_frame: OpenCV BGR 프레임
            target_width: 목표 표시 너비
            target_height: 목표 표시 높이
            
        Returns:
            PhotoImage 객체
        """
        # BGR to RGB 변환
        frame_rgb = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # 크기 조정 (비율 유지)
        if target_width > 1 and target_height > 1:
            img_width, img_height = pil_image.size
            scale_w = target_width / img_width
            scale_h = target_height / img_height
            scale = min(scale_w, scale_h)
            
            if scale > 0 and scale != 1.0:
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                if new_width > 0 and new_height > 0:
                    pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        return ImageTk.PhotoImage(image=pil_image)

    def resize_with_padding(self, image, target_width, target_height):
        """
        이미지를 목표 크기에 비율 유지하면서 패딩 추가
        
        Returns:
            tuple: (padded_image, scale, pad_x, pad_y, resized_w, resized_h)
        """
        h, w = image.shape[:2]
        
        # 비율 유지를 위한 스케일 계산
        scale = min(target_width / w, target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 리사이즈
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 패딩 계산 (중앙 정렬)
        pad_x = (target_width - new_w) // 2
        pad_y = (target_height - new_h) // 2
        
        # 패딩 추가 (검은색)
        padded = cv2.copyMakeBorder(
            resized,
            pad_y, target_height - new_h - pad_y,  # top, bottom
            pad_x, target_width - new_w - pad_x,   # left, right
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        
        return padded, scale, pad_x, pad_y, new_w, new_h