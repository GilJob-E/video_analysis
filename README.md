# video_analysis


웹캠 영상에서 **시선(아이컨택), 미소, 끄덕임, 앞으로 숙이기** 등의 비언어적 특성을 추출하고  
면접 상황에서의 **시선 비율 / 표정 / 고개 끄덕임**을 정량적으로 분석하는 Python 스크립트입니다.



## 기반 아이디어

- Ye et al., **Low-cost Geometry-based Eye Gaze Detection using Facial Landmarks Generated through Deep Learning**  
  - 얼굴 랜드마크로부터 8차원 시선 기하 특성(눈동자 위치, 얼굴 크기, 얼굴 위치, head rotation 등)을 구성하고  
    이를 이용해 시선 방향을 추정하는 **Geometry-based Gaze** 아이디어를 차용했습니다.
- Acarturk et al., 2021, *Gaze Aversion in Conversational Settings: An Investigation Based on Mock Job Interview*  
  - 모의 면접 환경에서 시선 회피/아이컨택 비율을 정량 분석한 결과를 참고하여,  
    면접 상황에서의 **“평균적인 eye contact 비율”**과 그 **분산(표준편차)**에 대한 규범값을 가져왔습니다.
- Esmer, 2022, *An Experimental Investigation of Gaze Behavior Modeling in VR (MSc Thesis)*  
  - VR 환경에서의 시선 행동 모델링 실험을 기반으로, **시선 유지/회피 비율 분포**를 보정하는 데 참고했습니다.

본 스크립트는 위 아이디어 및 통계를 바탕으로,  
**“화면 중앙 한 점 캘리브레이션 → 프레임 단위 시선 기하 특성 계산 → 아이컨택 여부 및 비율 산출”** 파이프라인을 제공합니다.



## 설치

### 1) 의존성

```bash
pip install opencv-python mediapipe numpy
```

- Python >= 3.9 권장
- 웹캠(또는 가상 카메라) 필요

### 2) 파일 배치

```text
project/
├── face.py               # 이 README가 설명하는 스크립트
└── face_landmarker.task   # MediaPipe FaceLandmarker 모델 파일
```

`face.py`와 `face_landmarker.task`를 **같은 폴더**에 두어야 합니다.



## 빠른 시작

```bash
python face.py
```

실행하면 두 단계로 진행됩니다.

1. **중앙 한 점 캘리브레이션**
   - `Center Calibration` 창이 열립니다.
   - 화면 중앙의 초록색 점을 바라보고,
   - 준비되면 **키보드 `c`** 를 눌러 수집을 시작합니다.
   - 약 `num_samples` 프레임(기본 120프레임) 동안 자동으로 데이터를 모읍니다.
   - 언제든 **`q`** 키로 종료 가능.

2. **인터뷰 세션(Interview Session)**
   - `Interview (Eye Contact)` 창이 열립니다.
   - 웹캠을 바라보며 평소처럼 말하면,
     - 아이컨택 여부,
     - 아이컨택 비율,
     - z-score,
     - 평균 미소 강도,
     - 끄덕임 횟수
     등을 실시간으로 화면에 오버레이합니다.
   - **`q`** 키를 누르면 세션이 종료되고,  
     터미널에 전체 요약이 출력됩니다.



## Ye et al. 기반 Eye-contact 로직 흐름

Ye et al.의 논문은 **얼굴 랜드마크만으로 저비용 시선 추정을 수행**하기 위해, 다음과 같은 흐름을 제안합니다.

1. **얼굴 랜드마크 → Geometry Feature 추출**
   - 눈 주변/얼굴의 특정 랜드마크 좌표를 사용해,
     - 눈동자 위치 편차(PPx, PPy)
     - 얼굴 회전(RX: pitch, RY: yaw)
     - 얼굴 크기(Hf, Wf)
     - 얼굴의 이미지 내 위치(PME_x, PME_y)
   - 총 8차원 벡터로 구성된 **Geometry-based Gaze Feature**를 만듭니다.

2. **정면(eye contact) 상태에 대한 기준 분포 학습**
   - 사람이 **정면(또는 특정 타깃)을 응시하는 상태**에서 위 8D feature를 여러 샘플로 수집합니다.
   - 이 분포를 이용해
     - 평균 벡터 μ (8차원)
     - 표준편차 벡터 σ (8차원)
   을 구하고, “정면 응시” 상태의 **기준 영역**을 정의합니다.

3. **새 프레임에 대한 z-score 및 편차 계산**
   - 새로운 프레임에서 얻은 feature `f`에 대해
     - `z = (f - μ) / σ` (각 차원별 z-score)
   - 그 중에서 **눈/머리 관련 차원**들(PPx, PPy, RX, RY)의 길이를 사용해
     - `dev_eye  = sqrt(z_PPx^2 + z_PPy^2)`
     - `dev_head = sqrt(z_RX^2  + z_RY^2)`
   - 즉, “정면 기준 분포에서 얼마나 떨어져 있는지”를 측정합니다.

4. **Threshold를 이용한 Eye-contact 판정**
   - Ye et al.의 아이디어를 따라,
     - dev_eye, dev_head가 설정한 임계값(thr_eye, thr_head) 안에 있으면  
       → **정면 응시(eye contact)** 로 판단
     - 임계값을 넘어가면  
       → 시선 회피(또는 다른 곳 응시)로 간주
   - 본 구현에서는 이 논리를 **Rule-based 방식**으로 단순화하여 사용합니다.

우리 구현의 `eye_contact_from_features_ye` 함수는 위 흐름을 그대로 따라가되,  
Ye et al.의 분류기를 대신해 **편차 기반 점수(final_score)** 를 계산하여 연속적인 0~1 스코어로 사용합니다.



## 추출 특성

### Target Features (세션 요약에 출력되는 핵심 지표)

| 특성               | 설명                                                 | 카테고리    |
|--------------------|------------------------------------------------------|------------|
| `eye_contact_ratio`| 얼굴이 검출된 프레임 중 아이컨택으로 판정된 비율(0~1) | Gaze       |
| `z_eye`            | 외부 통계(μ=0.70, σ=0.15) 기준 eye_contact_ratio Z-Score | Gaze   |
| `avg_smile_0_100`  | 세션 동안 평균 미소 강도 (0~100, Duchenne smile 기반) | Expression |
| `nod_count`        | 면접 중 고개를 끄덕인 횟수                           | Head Movement |

> 🧮 eye-contact 기준 통계  
> `EYE_CONTACT_MEAN_RATIO = 0.70`  
> `EYE_CONTACT_STD_RATIO  = 0.15`  
> 위 값은 **모의 면접 및 VR 기반 시선 연구(Acarturk et al., 2021; Esmer, 2022)**에서 보고된  
> 평균적인 시선 유지/회피 비율 분포를 참고하여 설정한 규범값입니다.



### 전체 추출 특성

#### 1) 8D Geometry Gaze Feature (Ye 스타일, 프레임 단위)

`compute_gaze_features_ye(landmarks)` → 길이 8짜리 벡터:

| 인덱스 | 이름     | 설명                                         |
|--------|----------|----------------------------------------------|
| 0      | `PPx`    | 양 눈 pupil 위치의 수평 편차 (얼굴 너비 정규화) |
| 1      | `PPy`    | 양 눈 pupil 위치의 수직 편차 (얼굴 높이 정규화) |
| 2      | `RX`     | 얼굴 상하 회전 관련 head pitch proxy        |
| 3      | `RY`     | 얼굴 좌우 회전 관련 head yaw proxy          |
| 4      | `Hf`     | 얼굴 높이(3D 거리, 눈-코 사이)              |
| 5      | `Wf`     | 얼굴 너비(3D 거리, 좌/우 눈 안쪽 꼬리 사이) |
| 6      | `PME_x`  | 이미지 내 얼굴 중앙 x 좌표                  |
| 7      | `PME_y`  | 이미지 내 얼굴 중앙 y 좌표                  |

캘리브레이션 단계에서 이 벡터들의 **평균/표준편차**를 저장하고,  
인터뷰 세션에서는 `(feat - mean) / std` 를 이용해 중앙에서의 편차를 계산합니다.



#### 2) 아이컨택 지표 (프레임 단위 구현)

`eye_contact_from_features_ye` 내부 로직 개요:

- 입력:
  - `feat` : 현재 프레임의 8D feature  
  - `mean_feat`, `std_feat` : 캘리브레이션에서 얻은 평균/표준편차  

```text
z = (feat - mean_feat) / std_feat

z_PPx = z[0]
z_PPy = z[1]
z_RX  = z[2]
z_RY  = z[3]

dev_eye  = sqrt(z_PPx^2 + z_PPy^2)
dev_head = sqrt(z_RX^2  + z_RY^2)

is_contact = (dev_eye <= thr_eye) and (dev_head <= thr_head)
score_eye  = max(0, 1 - dev_eye / thr_eye)
score_head = max(0, 1 - dev_head / thr_head)

final_score = 0.7 * score_eye + 0.3 * score_head
```

| 특성        | 설명                                      |
|-------------|-------------------------------------------|
| `is_contact`| 해당 프레임이 아이컨택(true)인지 여부     |
| `score`     | 0~1 사이의 연속적인 아이컨택 점수         |
| `dev_eye`   | 눈동자 편차 (중앙에서 얼마나 벗어났는지)  |
| `dev_head`  | 머리 회전 편차                            |



#### 3) 표정 / 미소

`smile(blendshapes)`:

- MediaPipe 블렌드셰이프 사용:
  - `mouthSmileLeft`, `mouthSmileRight`
  - `eyeSquintLeft`, `eyeSquintRight`
- 기본 미소 + 눈찡그림(Duchenne smile)을 결합해 0~1 점수 생성

세션 요약:

- `avg_smile_0_1` : 전체 face frame에 대한 평균 미소 점수
- `avg_smile_0_100 = avg_smile_0_1 * 100.0`



#### 4) 머리 움직임 (끄덕임 / 앞으로 숙이기)

- **끄덕임(nod)**: `detect_nod(pitch_list)`
  - 얼굴 pitch 이력을 보고,
  - 일정 임계값을 기준으로 “위를 보다가 아래로 숙이고 다시 올리는” 패턴을 감지

사용되는 상수 예시:

```python
DOWN_TH = -0.10
UP_TH   = -0.05
MAX_NOD_FRAMES = 240  # 약 1초(30fps 기준)
```

- **앞으로 숙이기(lean forward)**:  
  - `detect_lean_forward(prev_nose_z, cur_nose_z, move_th=0.02)`
  - 코의 z 값이 카메라 쪽으로 일정 거리 이상 가까워졌을 때 True  
  - (현재 스크립트에서는 통계 출력보다 이벤트 감지에 중점)


## API 사용법 (모듈 import 활용)

스크립트를 단순 실행하는 것 외에, 모듈로 import해서도 사용할 수 있습니다.

## 1. Dependencies

### 1.1 Python Version

- Python **3.9 이상** 권장

### 1.2 Required Packages

`face.py`에서 사용하는 외부 패키지:

```python
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
```

따라서 다음 패키지가 필요합니다.

```bash
pip install opencv-python mediapipe numpy
```

### 1.3 Model File

- MediaPipe Face Landmarker 모델 파일:

```text
face_landmarker.task
```

- 파일 위치:
  - `face.py`와 **같은 폴더**에 두어야 합니다.

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "face_landmarker.task")

if not os.path.exists(model_path):
    raise FileNotFoundError("'face_landmarker.task' 파일을 찾을 수 없습니다. 이 파일과 같은 폴더에 두세요.")
```

---

## 2. Basic API Usage

`face.py`는 크게 세 단계를 위한 API를 제공합니다.

1. Face Landmarker 초기화
2. 중앙 한 점 캘리브레이션
3. 인터뷰 세션(아이컨택/미소/끄덕임 추적)

### 2.1 모듈 임포트

```python
from face import (
    build_face_landmarker,
    run_center_calibration,
    run_interview_session,
    detect_landmarks,
    compute_gaze_features_ye,
    eye_contact_from_features_ye,
    smile,
    detect_nod,
    detect_lean_forward,
    head_pose_matrix,
    matrix_to_euler,
    get_blendshape_map
)
```

> 실제 파일 이름이 `face.py`가 아니라면, 해당 파일명으로 변경해서 import 해야 합니다.

---

### 2.2 Face Landmarker 초기화

```python
detector = build_face_landmarker()
```

- MediaPipe `FaceLandmarker`를 **VIDEO 모드**로 생성합니다.
- 옵션:
  - `num_faces=1`
  - `min_face_detection_confidence=0.5`
  - `min_face_presence_confidence=0.5`
  - `min_tracking_confidence=0.5`
  - `output_face_blendshapes=True`
  - `output_facial_transformation_matrixes=True`

---

### 2.3 중앙 한 점 캘리브레이션

사용자가 화면 중앙의 점을 바라보도록 하고,  
Ye 스타일 8D geometry feature의 평균/표준편차를 추정합니다.

```python
mean_feat, std_feat = run_center_calibration(detector, num_samples=120)
```

- `num_samples`:
  - 수집할 프레임 수 (기본 120 ≈ 4초 @30fps)
- 반환값:
  - `mean_feat`: 8D feature 평균 (numpy array, shape `(8,)`)
  - `std_feat` : 8D feature 표준편차 (numpy array, shape `(8,)`)

---

### 2.4 인터뷰 세션 실행

캘리브레이션 결과(`mean_feat`, `std_feat`)를 사용해  
실시간으로 아이컨택/미소/끄덕임을 추적합니다.

```python
run_interview_session(
    detector,
    mean_feat,
    std_feat,
    thr_h=2.0,  # 눈 관련 z-score threshold
    thr_v=4.0,  # 머리 회전 z-score threshold
)
```

- 웹캠을 열어 프레임 단위로:
  - 얼굴 랜드마크, 블렌드셰이프, head pose 추출
  - Ye 8D feature 계산 → 중앙 기준 z-score → eye-contact 판정
  - Duchenne smile 스코어 계산
  - pitch 이력으로 nod 감지
- 화면 오버레이 정보:
  - Eye-contact ratio
  - Eye-contact score
  - z-eye score
  - 평균 Smile intensity
  - Nod Count
  - Pitch 값
- 종료:
  - `q` 키를 누르면 세션 종료 후 요약 통계 출력

---

### 2.5 프레임 단위 API 사용 예시

#### 2.5.1 얼굴 탐지 + 랜드마크/표정/자세

```python
import cv2
from face import build_face_landmarker, detect_landmarks

detector = build_face_landmarker()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks, blendshapes, pitch, yaw = detect_landmarks(detector, frame)

    if landmarks is not None:
        # landmarks: MediaPipe face_landmarks
        # blendshapes: 표정 블렌드셰이프
        # pitch, yaw: 라디안 단위 머리 회전
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

#### 2.5.2 8D Gaze Feature + Eye-contact 수동 계산

```python
import numpy as np
from face import (
    compute_gaze_features_ye,
    eye_contact_from_features_ye,
)

# 캘리브레이션에서 얻은 mean_feat, std_feat
mean_feat = ...
std_feat = ...

feat = compute_gaze_features_ye(landmarks)  # landmarks는 MediaPipe 결과

is_contact, score, dev_eye, dev_head = eye_contact_from_features_ye(
    feat,
    mean_feat,
    std_feat,
    thr_eye=2.0,
    thr_head=4.0,
)

print("Eye-contact:", is_contact, "score=", score)
```

---

#### 2.5.3 Smile / Nod / Lean Forward 단독 사용

```python
from face import smile, detect_nod, detect_lean_forward

# 1) Smile
smile_score = smile(blendshapes)  # 0 ~ 1
print("Smile intensity:", smile_score)

# 2) Nod
pitch_history = []
pitch_history.append(pitch)  # 매 프레임 pitch 추가

if detect_nod(pitch_history):
    print("Detected nod!")

# 3) Lean forward
prev_nose_z = None
cur_nose_z = nose_landmark.z  # 프레임마다 업데이트

if detect_lean_forward(prev_nose_z, cur_nose_z, move_th=0.02):
    print("Leaning forward detected!")
prev_nose_z = cur_nose_z
```

---

## 3. CLI Usage

`face.py`를 직접 실행하면 다음 순서로 동작합니다.

```bash
python face.py
```

1. `build_face_landmarker()` 호출
2. `run_center_calibration(detector, num_samples=120)` 실행
3. `run_interview_session(detector, mean_feat, std_feat, thr_h=2.0, thr_v=1.5)` 실행

웹캠 창 2개가 순서대로 뜨며,  
- 첫 번째 창: **Center Calibration**
- 두 번째 창: **Interview (Eye Contact)**  
q 키로 각 단계 종료가 가능합니다.




## 정규화 (Normalization)

### 1) Gaze Feature Z-Score (중앙 기준)

- 캘리브레이션 단계:

```python
feats = np.stack(feats, axis=0)  # (N, 8)

mean_feat = feats.mean(axis=0)
std_feat  = feats.std(axis=0, ddof=0)
std_feat  = np.where(std_feat < 1e-6, 1e-6, std_feat)  # 0 분산 방지
```

- 인터뷰 세션에서 각 프레임:

```python
z = (feat - mean_feat) / std_feat
```

이 값을 바탕으로 **dev_eye / dev_head**를 계산해  
“중앙에서 얼마나 벗어났는지”를 평가합니다.

### 2) Eye-contact Ratio Z-Score (외부 통계 기반)

인터뷰가 끝난 후 전체 요약에서:

```python
eye_ratio = eye_contact_frames / total_face_frames
z_eye = (eye_ratio - EYE_CONTACT_MEAN_RATIO) / EYE_CONTACT_STD_RATIO
# EYE_CONTACT_MEAN_RATIO = 0.70
# EYE_CONTACT_STD_RATIO  = 0.15
```

- `z_eye > 0` : 평균보다 더 많이 카메라를 봄  
- `z_eye ≈ 0` : 평균적인 수준  
- `z_eye < 0` : 평균보다 적게 카메라를 봄  



## 프로젝트 구조

현재는 단일 스크립트 구조입니다.

```text
.
├── face.py               # 메인 스크립트 (캘리브레이션 + 인터뷰 세션)
└── face_landmarker.task   # MediaPipe FaceLandmarker 모델
```



## 한계점

1. **환경 민감도**
   - 카메라 위치, 해상도, 조명, 화면 크기에 따라 캘리브레이션/임계값이 달라질 수 있습니다.
2. **파라미터 해석**
   - `DOWN_TH = -0.10` `UP_TH   = -0.05`는 직접 테스트한 결과이고,
   - smile의 z-score를 계산할 때 사용 한 값은 kaggle의 First Impressions V2 (CVPR'17) - Training 데이터셋을 일부 사용
3. **의미가 있는가**
   - 대답만을 하는 영상을 입력으로 받으면 smile, nod가 대답과 동시에 이루어질 수 있는지?



## 참고 문헌

- Acarturk, C. et al. (2021). *Gaze Aversion in Conversational Settings: An Investigation Based on Mock Job Interview.*  
- Esmer, B. (2022). *An Experimental Investigation of Gaze Behavior Modeling in VR.* MSc Thesis.  
- Ye, X. et al. *Low-cost Geometry-based Eye Gaze Detection using Facial Landmarks Generated through Deep Learning.*  
- MediaPipe Face Landmarker (Face mesh & blendshapes & transformation matrix).
