
## 기존 opencv 설치 제거

```bash
pip uninstall opencv-python
```

## contrib 모듈 설치

```bash
pip install opencv-contrib-python
```

## CSRT 추적기
```python
import cv2
tracker = cv2.TrackerCSRT_create()



```

cv2.TrackerCSRT_create()는 OpenCV 라이브러리에서 제공하는 객체 추적기(object tracker)를 생성하는 함수입니다. CSRT는 "Channel and Spatial Reliability Tracker"의 약자로, 비교적 정확도가 높은 추적 알고리즘 중 하나로 알려져 있습니다.

주요 특징 및 설명은 다음과 같습니다:

목적: 비디오 시퀀스에서 특정 객체의 움직임을 따라가는 데 사용됩니다. 초기 프레임에서 추적할 객체의 위치(보통 바운딩 박스 형태)를 지정해주면, 다음 프레임들에서 해당 객체의 위치를 예측하여 반환합니다.
CSRT 알고리즘:
이름에서 알 수 있듯이 "채널 신뢰도"와 "공간 신뢰도"를 활용하여 추적합니다.
Discriminative Correlation Filter (DCF)를 기반으로 하며, 공간적 신뢰도 맵(spatial reliability map)을 사용하여 필터가 이미지의 특정 영역에만 집중하도록 합니다. 이를 통해 배경 정보나 노이즈가 많은 영역의 영향을 줄여 더 정확한 추적을 가능하게 합니다.
HoG (Histogram of Oriented Gradients)와 Color Names 같은 표준 특징들을 사용합니다.
성능:
정확도: CSRT는 OpenCV에서 제공하는 다른 추적기들(예: KCF, MOSSE)에 비해 일반적으로 높은 정확도를 보입니다. 객체의 회전, 크기 변화, 그리고 어느 정도의 가려짐(occlusion) 상황에서도 비교적 강인한 추적 성능을 제공합니다.
속도: 높은 정확도를 제공하는 대신, 다른 일부 추적기들(특히 MOSSE나 KCF)보다 속도가 느릴 수 있습니다. 따라서 실시간성이 매우 중요한 애플리케이션보다는 정확도가 우선시될 때 더 적합할 수 있습니다.




