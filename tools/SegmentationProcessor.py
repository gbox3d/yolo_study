import cv2
import numpy as np
from ultralytics import SAM
import threading

class SegmentationProcessor:
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
        return self._model_loaded_successfully and self.sam_model is not None

    def process_image_sync(self, image, prompt_data, prompt_type="point"):
        """
        (동기 방식) 이미지에서 세그멘테이션 수행. 기존 process_image 로직.
        """
        if not self.is_model_loaded():
            # print("SAM 모델이 로드되지 않아 처리를 건너뜁니다.")
            raise RuntimeError("SAM model not loaded.")

        try:
            h_img, w_img = image.shape[:2]
            results = None
            if prompt_type == "point":
                point_prompt = prompt_data
                # print(f"SAM 처리 (point prompt): {w_img}x{h_img} 이미지, 포인트 {point_prompt}")
                results = self.sam_model(image, points=[point_prompt], labels=[1])
            elif prompt_type == "bbox":
                bbox_prompt = prompt_data
                x, y, w, h = bbox_prompt
                sam_formatted_bbox = [x, y, x + w, y + h]
                # print(f"SAM 처리 (bbox prompt): {w_img}x{h_img} 이미지, BBox {sam_formatted_bbox}")
                results = self.sam_model(image, bboxes=[sam_formatted_bbox], labels=[1])
            else:
                raise ValueError(f"지원되지 않는 prompt_type: {prompt_type}")
            
            if results and results[0].masks is not None and len(results[0].masks.data) > 0:
                mask = results[0].masks.data[0].cpu().numpy().astype(bool)
                mask_uint8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                bbox = None
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    bx, by, bw, bh = cv2.boundingRect(largest_contour)
                    bbox = [bx, by, bw, bh]
                return mask, contours, bbox
            else:
                # print("SAM 결과에서 마스크를 찾을 수 없습니다.")
                return None, [], None
        except Exception as e:
            # print(f"SAM 동기 처리 중 오류: {e}")
            raise e # 오류를 호출자에게 전파

    def _process_image_worker(self, image_copy, prompt_data, prompt_type, callback, callback_kwargs):
        """ (워커 스레드) 실제 세그멘테이션 작업 및 콜백 호출 """
        error = None
        mask, polygons, bbox = None, [], None
        try:
            mask, polygons, bbox = self.process_image_sync(image_copy, prompt_data, prompt_type)
        except Exception as e:
            error = e
        
        if callback:
            # 콜백은 워커 스레드에서 직접 호출됨.
            # LabelingTabUI는 이 콜백 내에서 app.master.after를 사용해야 함.
            callback(mask, polygons, bbox, error, **callback_kwargs)

    def process_image_async(self, image, prompt_data, prompt_type, callback, **callback_kwargs):
        """
        (비동기 방식) 세그멘테이션을 별도 스레드에서 수행하고 완료 시 콜백 호출.
        callback_kwargs는 콜백 함수에 그대로 전달되어 호출 컨텍스트를 구분하는 데 사용됨.
        """
        if not self.is_model_loaded():
            # 모델 미로드 시 즉시 콜백 (에러 전달) - 일관성을 위해 이것도 스레드에서 호출되도록 함
            err_args = (None, [], None, RuntimeError("SAM model not loaded."))
            threading.Thread(target=callback, args=err_args, kwargs=callback_kwargs, daemon=True).start()
            return

        image_copy = image.copy() # 스레드 간 데이터 공유 문제 방지

        # 각 비동기 호출은 자체 스레드를 가짐
        thread = threading.Thread(
            target=self._process_image_worker,
            args=(image_copy, prompt_data, prompt_type, callback, callback_kwargs),
            daemon=True # 메인 스레드 종료 시 자동 종료
        )
        thread.start()