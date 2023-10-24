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

model : 모델정의파일 경로지정, pt는 전이학습과 훈련재계(resume) 에 사용, 모델정의파일(.yaml)을 넣어주면 처음부터 학습한다.    
```bash
yolo detect train data=coco128.yaml model=last.pt resume=True #학습재계
```

pretrained :  미세조정

```bash 
yolo detect train data=coco128.yaml model=modified_yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
```

### 학습 명령어의 예  

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=1000 imgsz=640 patience=200 batch=32
yolo detect train data=./madang.yaml model=yolov8n.pt epochs=1000 imgsz=640 batch=64 patience=200
yolo detect train data=./madang.yaml model=yolov8n.yaml epochs=1000 imgsz=640 batch=64 patience=200


yolo segment train data=data.yaml model=yolov8n-seg.yaml epochs=100 imgsz=640 batch=64 patience=200  pretrained=yolov8n-seg.pt
yolo segment train data=data.yaml model=yolov8s-seg.yaml epochs=100 imgsz=640 batch=32 patience=200  pretrained=yolov8s-seg.pt
yolo segment train data=data.yaml model=yolov8s-seg.yaml epochs=100 imgsz=640 batch=16 patience=200  pretrained=yolov8s-seg.pt
yolo segment train data=data.yaml model=yolov8s-seg.yaml epochs=100 imgsz=640 batch=16 patience=200  pretrained=yolov8s-seg.pt
yolo segment train data=data.yaml model=yolov8x-seg.yaml epochs=1000 imgsz=640 batch=4 patience=200
```

## model 정의 파일

훈련실행시에 model 파라메터에 정의하는 파일이다.  
yolov8.yaml 파일은 모델의 구성을 정의 하는 파일이다.  
모델의 아키텍쳐를 튜닝하고 싶을때 이 파일을 수정한다.  

## dataset 정의 파일

훈련실행시에 data 파라메터에 정의하는 파일이다.  
coco.yaml 은 데이터셋의 구성을 정의하는 대표적인 예제 파일이다.

```yaml
path: ../datasets/coco  # dataset root dir
train: train.txt  # train images (relative to 'path') 
val: val.txt   # val images (relative to 'path')
test:  test.txt # test images (relative to 'path')
```
path는 데이터셋의 루트 디렉토리를 의미한다.  
train은 학습데이터셋의 이미지가 있는 디렉토리 또는 리스트가 담겨있는 파일을 정의한다.  

train.txt 파일은 다음과 같은 구성을 가진다.  
```txt
./images/000000000139.jpg
./images/000000000285.jpg
./images/000000000632.jpg
....
```

라벨은 labels 디렉토리의 같은 이름의 디랙토리에 같은 이름의 파일로 저장된다.   
이렇게 하면 라벨파일은 ./images/000000000139.jpg 의 경우에는 ./labels/000000000139.txt 로 매칭된다.  

## 참고자료
https://github.com/ultralytics/ultralytics
