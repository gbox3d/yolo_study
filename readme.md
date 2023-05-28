# YOLO study 
영상인식AI 개발자 양성과정 YOLO 강의 교재  

**강의문의 : gbox3d@gmail.com**   


## 개발경 설치

```bash

pip install opencv-contrib-python

```

## 실습용 데이터셋 받는 방법

```bash
mkdir datasets
cd datasets
wget https://ultralytics.com/assets/coco128.zip
wget https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/datasets/coco128.yaml
unzip coco128.zip
``` 

## 실습용 데이터셋 받는 방법

```bash
python yolo_format_test.py --basepath /home/gbox3d/work/dataset/test/ --imageFile 150759820_900615604088631_2924655063235727439_n.jpg --data ./datasets/coco128.yaml
```