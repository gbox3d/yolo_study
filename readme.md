# yolo v5 tutorial for pytorch

## 필수 모듈 설치 

```sh
#우분투의경우 반드시 제일 먼저 설치한다.
sudo apt install liblzma-dev

pip install sklearn

```

* 라즈베리파이  
```sh
pip3 install PyYAML scipy tqdm numpy

```


## 가상 환경 셋업

pyenv, conda둘중 하나를 선택할수있다.
### pyenv 
```
sudo apt install liblzma-dev # 우분투에서 설치안되어있다면
pyenv install 3.8.6 # (만약 파이썬 버전이 없다면)
pyenv virtualenv 3.8.6 yolov5_pt

```

### ananconda
```
conda create -n yolo_pt python=3.8
```

## 모듈 설치 
```
pip install -r requirements.txt
```

## 가중치 파일 다운받기
```
cd models
./download_weights.sh
```


## training 

예제 data set 다운받기 
```
cd dataset
mkdir pistol
cd pistol

curl -L "https://public.roboflow.ai/ds/WKkUorQ71T?key=wIBAdyawPa" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

훈련시작
```
python train.py --img 416 --batch 16 --epochs 50 --data ../../dataset/pistol/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name gun_yolov5s_results
```

테스트  
test_conf.yaml 을 다음과 같이 작성한다.
```yaml
dataset_path : '/home/gbox3d/work/dataset/pistol/'
model_path : '/home/gbox3d/work/visionApp/yolov5/runs/train/gun_yolov5s_results/weights/best.pt'
test_index : 3
```
test_train.py를 실행시킨다.


## 참고자료
https://github.com/ultralytics/yolov5 
