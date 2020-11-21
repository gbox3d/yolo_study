# yolo v5 tutorial for pytorch

## 가상 환경 셋업

pyenv, conda둘중 하나를 선택할수있다.
### pyenv 
```
sudo apt install liblzma-dev # 우분투에서 설치안되어있다면
pyenv install 3.7.9 # (만약 파이썬 버전이 없다면)
pyenv virtualenv 3.7.9 yolo_pt

```

### ananconda
```
conda create -n yolo_pt python=3.7
```

## 모듈 설치 
pip install -r requirements.txt

## 참고자료
https://github.com/ultralytics/yolov5 
