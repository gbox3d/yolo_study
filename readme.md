# yolo v5 tutorial for pytorch

## 필수 모듈 설치 

```
sudo apt install liblzma-dev

pip install sklearn

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
pip install -r requirements.txt


## training data 다운받기 

```
cd dataset
mkdir pistol
cd pistol

curl -L "https://public.roboflow.ai/ds/WKkUorQ71T?key=wIBAdyawPa" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

## 참고자료
https://github.com/ultralytics/yolov5 
