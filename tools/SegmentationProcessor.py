import cv2
import numpy as np
from ultralytics import SAM

class SegmentationProcessor:
    """
    SAM 모델을 사용한 세그멘테이션 처리 클래스
    """
    def __init__(self, model_path="sam2_s.pt"):
        self.sam_model = None
        self._model_loaded_successfully = False
        
        try:
            self.sam_model = SAM(model_path)
            print(f"SAM 모델 로드 성공: {model_path}")
            self._model_loaded_successfully = True
        except Exception as e:
            print(f"SAM 모델 로드 실패: {e}")
            self.sam_model = None

    def is_model_loaded(self):
        """모델 로드 상태 확인"""
        return self._model_loaded_successfully and self.sam_model is not None

    def process_image(self, image, click_point):
        """
        이미지에서 세그멘테이션 수행
        
        Args:
            image: BGR 이미지 (임의 크기)
            click_point: [x, y] - 이미지 좌표계에서의 클릭 포인트
            
        Returns:
            tuple: (mask, contours, bbox)
                - mask: bool 마스크 (입력 이미지와 같은 크기)
                - contours: 폴리곤 목록
                - bbox: [x, y, w, h] 바운딩 박스
        """
        if not self.is_model_loaded():
            print("SAM 모델이 로드되지 않아 처리를 건너뜁니다.")
            return None, [], None

        try:
            h, w = image.shape[:2]
            x, y = click_point
            print(f"SAM 처리: {w}x{h} 이미지, 클릭 ({x}, {y})")
            
            # SAM 추론 실행
            results = self.sam_model(image, points=[click_point], labels=[1])
            
            if results and results[0].masks is not None and len(results[0].masks.data) > 0:
                # 마스크 추출
                mask = results[0].masks.data[0].cpu().numpy().astype(bool)
                print(f"마스크 생성 완료: {mask.shape}")
                
                # 컨투어 추출
                mask_uint8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                bbox = None
                if contours:
                    print(f"컨투어 {len(contours)}개 발견")
                    # 가장 큰 컨투어의 바운딩 박스
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    bbox = [x, y, w, h]
                    print(f"바운딩 박스: {bbox}")
                
                return mask, contours, bbox
            else:
                print("SAM 결과에서 마스크를 찾을 수 없습니다.")
                return None, [], None
                
        except Exception as e:
            print(f"SAM 처리 중 오류: {e}")
            return None, [], None