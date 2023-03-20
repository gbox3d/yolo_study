# Yolov8 Tutorial

## setup

```bash
pip install ultralytics
```

## train


먼저 데이터셋디랙토리로 이동한다.   

batch : 배치싸이즈  
epochs : 학습횟수  
patience : early stopping  
imgsz : 이미지 사이즈  
data : 데이터셋 경로  
model : 모델 경로, pt는 전이학습시 사용 yaml 처음부터 학습시 사용     

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=1000 imgsz=640 patience=200 batch=32
yolo detect train data=./madang.yaml model=yolov8n.pt epochs=1000 imgsz=640 batch=64 patience=200
yolo detect train data=./madang.yaml model=yolov8n.yaml epochs=1000 imgsz=640 batch=64 patience=200


yolo segment train data=data.yaml model=yolov8n-seg.yaml epochs=100 imgsz=640 batch=64 patience=200  pretrained=yolov8n-seg.pt
yolo segment train data=data.yaml model=yolov8s-seg.yaml epochs=100 imgsz=640 batch=32 patience=200  pretrained=yolov8s-seg.pt
yolo segment train data=data.yaml model=yolov8s-seg.yaml epochs=100 imgsz=640 batch=16 patience=200  pretrained=yolov8s-seg.pt
```



## 참고자료
https://github.com/ultralytics/ultralytics
