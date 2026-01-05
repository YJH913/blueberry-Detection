## JFST-DETR: Small-Target Fruit Detector (RT-DETR 기반)

### 1. 개요

**JFST-DETR**는 RT-DETR(Real-Time Detection Transformer)를 기반으로, 과수원과 같은 복잡한 농업 환경에서 **32×32 픽셀 이하의 소형 타겟**(열매, 병해충 등)을 안정적으로 검출하기 위해 설계된 모델입니다.  
이 레포의 코드는 논문 구조를 따라 다음 모듈을 모두 포함하도록 구현되어 있습니다.

- **Backbone**: ResNet-18 (기본값: **from scratch**)
- **SEPN (Spatial Enhancement Pyramid Network)**  
  - **SCM (Spatial Coding Module)**: Space-to-depth 연산으로 저해상도에서 잃기 쉬운 미세 공간 정보를 깊이 방향으로 재배치해 유지  
  - **GAAM (Global Awareness Adaptive Module)**: 대규모/전역/지역 분기를 결합하여 다중 스케일 타겟에 강건한 표현 학습
- **PSConv (Pinwheel-Shaped Convolution)**: 바람개비 형태 수용 영역을 갖는 비대칭 컨볼루션으로 복잡한 배경 노이즈를 억제
- **DySample (Dynamic Sampling)**: 단순 bilinear 보간 대신, 동적 오프셋과 가중치를 학습해 업샘플링 위치를 조정
- **Hybrid Encoder (RT-DETR 스타일)**: AIFI(Intra-scale Transformer) + CCFM(Cross-scale CNN)로 피처 맵을 혼합
- **RT-DETR Detection Head**: IoU-aware score head + Top-K query 선택 + Transformer Decoder + class/box head
- **Hungarian Matcher + SetCriterion**: DETR 계열과 동일하게 Hungarian 매칭 + Focal 수준의 CE/L1/GIoU 조합 손실


### 2. 아키텍처 상세

- **Backbone (`ResNet18Backbone`)**
  - 기본값 `backbone_pretrained: false` → **ImageNet 사전학습 없이** ResNet-18을 처음부터 학습
  - stage2~5 출력(`P2`~`P5`)을 neck으로 전달

- **SEPN (`SEPN`)**
  - 입력: `P2, P3, P4, P5`
  - **SCM**: `P2`에 space-to-depth 를 적용하여 채널을 4배로 확장 후, 3×3 conv 블록으로 정제
  - **GAAM**: 3단계(`P3`, `P4`, `P5`)에서 각각  
    - Large(대규모), Global(DCAM+FSAM), Local(depthwise conv) 분기를 계산  
    - 채널을 4-way로 분할(CSP 스타일) 후 concat + 1×1 conv로 융합
  - `P2`를 업샘플/다운샘플하여 `P3`/`P4`/`P5`와 결합, pyramid 상단으로 갈수록 더 풍부한 공간·주파수 정보를 보존

- **PSConv (`PSConv`)**
  - `P3`, `P4`, `P5` 각각에 대해 별도의 PSConv 블록을 적용 (`ps3`, `ps4`, `ps5`)
  - 좌/우/상/하 방향으로 비대칭 패딩 후 분기별 컨볼루션 → concat → 1×1 conv로 병합
  - 복잡한 배경에서의 경계 반응을 줄이고, 타겟 중심부 응답을 강화하는 역할

- **Dynamic Sampling (`DySample`)**
  - 채널 차원에서 offsets(2×r²)와 weight(r²)를 예측
  - 각 위치별로 r²개의 샘플링 포인트를 grid_sample로 추출 후, softmax weight로 가중합
  - 마지막에 pixel_shuffle을 통해 해상도를 r배로 확장  
  - 고정형 bilinear 업샘플에 비해 **소형 타겟 윤곽선과 세부 구조를 더 잘 복원**하도록 설계

- **Hybrid Encoder (`HybridEncoderBlock`)**
  - 각 스케일(`P3`, `P4`, `P5`)에 **AIFI** 적용 (Transformer Encoder 기반 intra-scale attention)
  - **CCFM**에서 세 스케일 피처를 P3 해상도 기준으로 결합 후, 다시 각 스케일 해상도로 분배
  - CNN과 Transformer의 장점을 결합한 RT-DETR 스타일 인코더

- **Detection Head (`RTDETRDetection`)**
  - Feature memory(B×N×C)에 대해:
    - IoU-aware score head로 각 위치 score 예측
    - 상위 `num_queries` 개 위치를 선택해 query로 사용(Top-K query selection)
    - Transformer Decoder를 통해 context-aware query 업데이트
    - 최종적으로 class logits(`num_classes + 1` with background) 및 bbox(cx,cy,w,h, Sigmoid)를 출력

- **Loss & Matching (`HungarianMatcher`, `SetCriterion`)**
  - `HungarianMatcher`:
    - class probability, bbox L1, generalized IoU를 조합해 cost matrix 구성
    - SciPy `linear_sum_assignment`로 최적 매칭 계산
  - `SetCriterion`:
    - Hungarian 매칭 결과로 실제 GT와 쿼리를 1:1 연결
    - **분류 손실**: CE + 배경에 대한 가중치 `eos_coef`(기본 0.1)를 적용해 no-object dominance 억제  
    - **bbox 손실**: L1 loss (cx,cy,w,h)  
    - **GIoU 손실**: generalized_box_iou 기반 1 - GIoU
    - 손실 가중치 기본값: `loss_ce: 2, loss_bbox: 5, loss_giou: 2`


### 3. 데이터셋 & 전처리

- **입력 형태**
  - 이미지: RGB, 기본 해상도 **640×640**으로 리사이즈
  - 마스크: 각 이미지에 대응하는 단일 채널 mask (`.png`)  
    - `cv2.findContours`로 외곽을 찾고, 각 contour에 대해 최소 외접 사각형 bbox 추출
    - bbox는 \([cx, cy, w, h]\) 형식으로 **이미지 크기로 정규화 (0~1)**  
    - 아주 작은 노이즈(너비/높이 < 3픽셀)는 제거

- **입력 정규화**
  - 픽셀: \([0,1]\) 범위로 스케일
  - 채널별 정규화:  
    \[
    x_\text{norm} = (x - \text{mean}) / \text{std}
    \]
    - `mean = [0.485, 0.456, 0.406]`
    - `std  = [0.229, 0.224, 0.225]`

- **라벨**
  - 현재 구현은 **단일 클래스**(예: 과실 vs 배경)를 가정
  - `labels` 텐서는 모두 0(단일 객체 클래스 ID), 배경은 DETR 스타일로 logits 마지막 index가 담당

- **데이터 분할 (기본)**
  - `config.yaml`의 비율에 따라:
    - **Train**: 70%
    - **Test**: 20%  (best 모델 선정 기준)
    - **Val**: 10%   (중간 학습 모니터링용)
  - 하나의 이미지 디렉토리와 마스크 디렉토리만 지정하면, 코드에서 무작위로 split

- **증강 (Albumentations, Train만)**  
  *지오메트리를 바꾸지 않는 photometric 변환만 사용해 bbox 일관성 유지*

  - RandomBrightnessContrast
  - RandomRain
  - RandomShadow
  - RandomSnow
  - ISONoise
  - RandomSunFlare
  - RandomFog


### 4. 훈련 레시피 & 환경

- **Optimizer**: **AdamW**
- **Learning Rate**
  - Backbone: `1e-5` (from scratch 안정성을 위한 보수적 세팅)  
  - Neck/Head 등 나머지: `1e-4` (논문 초기 학습률과 동일)
- **Epochs**: 기본 100 (논문에서도 약 90 epoch 부근에서 수렴)
- **Batch Size**:
  - `train.batch_size = 1`
  - `train.accum_steps = 4`  
  → **Effective batch size ≈ 4 (1×4)** → 논문에서 언급한 배치 사이즈와 일치
- **입력 해상도**: 640×640
- **Backbone Pretrain**: 기본값 **False** (ResNet-18 from scratch)
- **AMP (Automatic Mixed Precision)**:
  - `torch.cuda.amp.autocast` + `GradScaler`를 사용해 학습 속도와 메모리 사용량 최적화
  - Gradient accumulation 및 gradient clipping과 호환되도록 구현
- **Gradient Clipping**: `max_norm = 1.0`

**권장 환경 (논문 환경 기준 예시)**:

- Python 3.11.x
- PyTorch 2.2.x + CUDA 12.x
- NVIDIA GPU (예: RTX 4060 Laptop, 8GB VRAM 이상 권장)


### 5. 설정 파일 (`config.yaml`)

모든 하이퍼파라미터와 데이터 경로는 `config.yaml`로 관리합니다.  
필수로 수정해야 하는 항목은 보통 `data.img_dir`, `data.mask_dir` 정도입니다.

```yaml
seed: 42

data:
  img_dir: "/path/to/Images"
  mask_dir: "/path/to/Masks"
  img_size: 640
  train_ratio: 0.7
  test_ratio: 0.2
  val_ratio: 0.1

model:
  num_classes: 1
  hidden_dim: 128
  num_queries: 50
  backbone_pretrained: false

matcher:
  cost_class: 1
  cost_bbox: 5
  cost_giou: 2

loss:
  eos_coef: 0.1
  weights:
    loss_ce: 2
    loss_bbox: 5
    loss_giou: 2

optim:
  lr_backbone: 1.0e-5
  lr: 1.0e-4
  weight_decay: 1.0e-4

amp:
  enabled: true

train:
  epochs: 100
  batch_size: 1
  num_workers: 0
  accum_steps: 4
  clip_norm: 1.0

eval:
  score_thresh: 0.5
  iou_thresh: 0.5
  ckpt: ""   # --eval-only 에서 --ckpt 미지정 시 사용

save:
  dir: "./checkpoints"
  metric: "map50"  # "map50" 또는 "f1"
```


### 6. 설치 & 실행 방법

#### 6.1. 의존성 설치

프로젝트 루트(`JSFT_DETR`)에서:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# 추가로 PyTorch는 환경에 맞게 설치 (예: CUDA 12.x)
# https://pytorch.org 의 설치 가이드를 따르세요.
```

`requirements.txt`는 다음 주요 패키지를 포함합니다:

- **matplotlib**: 학습/평가 곡선 플로팅
- **pyyaml**: `config.yaml` 파싱
- **albumentations**: 데이터 증강


#### 6.2. 학습 (Training)

```bash
cd /home/jepetolee/PycharmProjects/blueberry-Detection/JSFT_DETR

python3 train.py --config config.yaml
```

실행 시:

- 현재 사용 중인 **GPU/CPU 정보** (이름, 메모리, compute capability)가 먼저 출력됩니다.
- 매 epoch마다 콘솔에 다음 정보가 출력됩니다.
  - Train loss (total / CE / bbox / GIoU)
  - GT per image, empty 이미지 수
  - Val (70/20/10 중 10% 부분) Precision / Recall / F1 / mAP@0.5
  - Test (20%) Precision / Recall / F1 / mAP@0.5 및 **BEST 여부**

체크포인트:

- `save.dir` (기본 `./checkpoints`)에 다음 파일이 생성됩니다.
  - `last.pth`: 최근 epoch 상태
  - `best.pth`: **Test set metric (기본 mAP@0.5 또는 F1)** 기준으로 가장 좋은 모델
  - `config.resolved.json`: 실험에 사용된 설정 기록
  - `history.csv`, `history.json`: epoch별 지표 기록
  - `history.png`: 학습/평가 곡선 플롯


#### 6.3. 평가만 수행 (Eval-only)

이미 학습된 체크포인트가 있을 때, Test set에서 성능만 측정하려면:

```bash
python3 train.py \
  --config config.yaml \
  --eval-only \
  --ckpt ./checkpoints/best.pth
```

또는 `config.yaml`의 `eval.ckpt` 항목을 채워 두고 `--ckpt`를 생략할 수도 있습니다.


### 7. 프로젝트 구조

```text
JSFT_DETR/
  ├─ train.py          # 학습/평가 진입점 (YAML config 기반)
  ├─ config.yaml       # 데이터/모델/훈련 설정
  ├─ jfst_detr.py      # JFST-DETR 전체 아키텍처 정의
  ├─ SEPN.py           # SEPN (SCM + GAAM) 및 ResNet-18 backbone
  ├─ DY_PS.py          # DySample, PSConv 모듈
  ├─ hybrid_encoder.py # Hybrid Encoder (AIFI + CCFM)
  ├─ Hungrian_match.py # HungarianMatcher, RTDETRDetection, SetCriterion
  ├─ MaskDataset.py    # 마스크 기반 bbox 생성 + 증강/정규화
  ├─ IOU_select.py     # (옵션) IoU 관련 유틸
  ├─ requirements.txt  # 필수 파이썬 패키지
  └─ README.md         # (현재 문서)
```


### 8. 참고 및 확장 아이디어

- **Backbone LR 조정**: 논문과 완전히 동일하게 맞추고 싶다면 `optim.lr_backbone`을 `1e-4`로 높여 전체 파라미터를 같은 학습률로 두는 것도 가능합니다.
- **다중 클래스 확장**: 다른 과실/병해충 등 복수 클래스로 확장하려면:
  - `config.yaml`의 `model.num_classes`를 변경
  - `MaskDataset`에서 `labels`를 실제 클래스 ID로 채우도록 수정
- **다른 데이터셋 적용**: Olive Fruit, CherryBBCH72, PDT 등 다른 소형 타겟 데이터셋에도 동일한 파이프라인을 적용할 수 있으며,  
  이때는 `data.img_dir` / `data.mask_dir` / 라벨 형식에 맞는 `MaskDataset` 구현만 적절히 조정하면 됩니다.


