import cv2
import numpy as np
from ultralytics import SAM
# from tkinter import messagebox # Tkinter 의존성 제거

class SegmentationProcessor:
    """
    SAM 모델을 사용하여 이미지 세그멘테이션을 처리하는 클래스.
    마스크, 폴리곤, 바운딩 박스를 생성합니다.
    """
    def __init__(self, model_path="sam2_s.pt", processing_width=640, processing_height=640):
        self.processing_width = processing_width
        self.processing_height = processing_height
        self.sam_model = None
        self._model_loaded_successfully = False # 모델 로드 성공 여부 플래그
        try:
            self.sam_model = SAM(model_path)
            print(f"SAM 모델 로드 성공 ('{model_path}')")
            self._model_loaded_successfully = True
        except Exception as e:
            print(f"SAM 모델('{model_path}')을 로드할 수 없습니다: {e}\n모델 파일이 올바른 경로에 있는지 확인하세요.")
            self.sam_model = None

    def is_model_loaded(self):
        return self._model_loaded_successfully and self.sam_model is not None

    def process_image(self, frame_bgr, click_point_processed):
        """
        주어진 프레임과 클릭 포인트를 사용하여 세그멘테이션을 수행합니다.
        처리된 마스크, 폴리곤, 바운딩 박스를 반환합니다.
        클릭 포인트는 이미 processing_width, processing_height 기준으로 변환된 값이어야 합니다.
        frame_bgr은 이미 processing_width, processing_height로 리사이즈된 상태여야 합니다.
        """
        if not self.is_model_loaded():
            print("SAM 모델이 로드되지 않아 처리를 건너<0xEB><0><0xA9>니다.")
            return None, [], None

        mask_bool_processed = None
        polygons_processed = []
        bbox_processed = None

        try:
            # print(f"SAM 추론 시작 ({self.processing_width}x{self.processing_height}): 포인트 ({click_point_processed[0]}, {click_point_processed[1]})")
            results = self.sam_model(frame_bgr, points=[click_point_processed], labels=[1])
            
            if results and results[0].masks is not None and len(results[0].masks.data) > 0:
                mask_bool_processed = results[0].masks.data[0].cpu().numpy().astype(bool)
                # print(f"SAM 마스크 생성됨 ({self.processing_width}x{self.processing_height}).")
                
                mask_uint8_processed = mask_bool_processed.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                polygons_processed = contours
                
                if polygons_processed:
                    # print(f"{len(polygons_processed)}개의 폴리곤 추출됨 ({self.processing_width}x{self.processing_height}).")
                    largest_contour = max(polygons_processed, key=cv2.contourArea)
                    bbox_processed = cv2.boundingRect(largest_contour) # (x, y, w, h)
                    # print(f"최대 면적 폴리곤의 BBox ({self.processing_width}x{self.processing_height} 기준): {bbox_processed}")
                else:
                    # print(f"마스크에서 폴리곤을 추출하지 못했습니다 ({self.processing_width}x{self.processing_height}).")
                    pass
            else:
                # print("SAM 추론 결과에서 마스크를 찾을 수 없습니다.")
                pass
        except Exception as e:
            print(f"SAM 처리 중 오류 발생: {e}")
            mask_bool_processed = None
            polygons_processed = []
            bbox_processed = None
            
        return mask_bool_processed, polygons_processed, bbox_processed
