# Roadmap: 1D CNN 2차 검증 고도화

## 목표

실제 낙상은 놓치지 않고(높은 recall), 침대에 눕기·빨리 앉기·물건 줍기 같은 하드 ADL 오탐은 줄이는 것.

## 현재 정확도를 막고 있는 원인

코드 분석으로 확인한 원인을 영향이 큰 순서로 정리했습니다.

1. CNN이 낙상 동작을 보지 못함
   AI 검증은 ABNORMAL 진입 후 약 10초가 지난 LEVEL1 시점에, 직전 30프레임만 보고 판단합니다.
   이 30프레임은 넘어지는 동작이 아니라 "누워서 정지" 구간이라, 바닥에 누운 것과 침대에 누운 것을 정적 자세만으로 구분해야 합니다.
2. 학습 라벨 노이즈
   `dataset_builder.py`는 URFD 낙상 영상의 모든 슬라이딩 윈도우를 Fall로 라벨링합니다. 서 있는 구간과 누운 구간도 포함됩니다.
   URFD가 제공하는 프레임 단위 라벨(-1/0/1)을 쓰지 않습니다.
3. 학습 입력과 운영 입력의 불일치
   - URFD는 30fps라 30프레임이 1초이지만, Jetson 운영에서는 2~4초입니다.
   - 운영 입력은 EMA 스무딩된 키포인트, 학습 입력은 원본입니다.
   - 미검출 관절 (0,0)이 골반 기준으로 바뀌면서 큰 이상치가 됩니다.
   - 스케일과 종횡비 정규화가 없습니다.
   - 정규화 코드가 3곳에 복제되어 있습니다. (Day 3에서 features.py로 통합)
4. AI가 실제 낙상 알람을 지울 수 있던 경로 (Day 1에서 해결)
   - 가중치 경로가 어긋나면 랜덤 모델로 판정했습니다.
   - 억제 후 쿨다운과 리셋 때문에 누운 사람이 다시 알람을 받지 못했습니다.
5. 액티브 러닝 자기강화 루프
   AI가 억제한 클립을 검수 없이 전부 ADL로 학습합니다. AI가 놓친 실제 낙상도 ADL로 강화됩니다.
6. 평가 체계 부재
   소규모 URFD val 정확도로 모델을 고르고, 이벤트 단위 recall이나 시간당 오탐을 재는 도구가 없습니다.

## 채택 규칙

Day 2 이후 모든 변경은 `replay_eval` 결과로 채택 여부를 정합니다.

- 낙상 이벤트 recall이 떨어지지 않을 것
- 하드 ADL의 시간당 오탐(FA/h)이 줄어들 것
- 창 단위 accuracy는 진단용으로만 사용

## 일정

### Day 1: 안전성과 테스트 기반 ✅

CNN이 실제 낙상 알람을 조용히 지우는 경로를 막았습니다. 분류기의 판정 경계는 바꾸지 않았습니다.

- [x] 룰 엔진을 torch/numpy/cv2 없이 import 가능하게 (테스트 가능성)
- [x] 가중치 경로 해석, 로드 실패 시 비활성화(fail-open), `p_fall` API
- [x] verifier 주입, `AI_MODE`(veto/shadow/off), `p_fall` 임계값 게이트, 이벤트에 AI 판정 기록
- [x] 억제 시 쿨다운 미소모, suppressed-watch, CNN과 무관한 승격(escalation)
- [x] 합성 시나리오 회귀 테스트 (`tests/`)
- [x] 다관점 리뷰 반영: 승격 시 현재 붕괴 자세 확인, 분류기 싱글턴 스레드 안전화, docker-compose 환경변수 전달

직접 확인할 것 (Jetson):

- [ ] 기존 가중치로 기동 후 로그에 `enabled: True`와 `weights_hash`가 찍히는지
- [ ] 가중치 폴더 이름을 잠시 바꾼 뒤 `classifier DISABLED` 로그가 나오고 낙상 시 level1이 정상 송출되는지
- [ ] `AI_MODE=shadow`로 낙상을 연출해 `values.ai.p_fall`이 기록되는지

### Day 2: 측정 체계 ✅

판정 로직은 바꾸지 않고, 이후 변경을 숫자로 비교할 수 있게 합니다.

- [x] `app.eval`: 이벤트 단위 replay 평가, 결과 비교와 채택 규칙, 합성 baseline
- [x] `evalset_tools` 누수 검사 강화, manifest/라벨 예시
- [x] 타임스탬프 원시 키포인트 버퍼, ABNORMAL 진입 시점 사건 구간 기록
- [x] 이벤트 sidecar, 억제 클립에 AI 판정 구간 포함, 실측 FPS
- [ ] [owner] 아래 녹화/라벨링

계획했던 내용:

- `app/eval/replay_eval.py`: 녹화된 `obs_*.jsonl`과 라벨을 받아 엔진을 오프라인으로 재생합니다.
  rules-only와 rules+AI를 비교하고, 이벤트 recall, 시간당 오탐, 알람 지연, 억제된 실제 낙상 수를 냅니다.
- 결과 비교 스크립트와 합성 baseline을 추가하고, eval manifest 예시를 버전 관리합니다.
- `evalset_tools`: train과 eval 사이 subject/session 누수를 error로 처리합니다.
- 타임스탬프가 붙은 원시 키포인트 버퍼(lost 프레임 포함)를 추가하고, ABNORMAL 진입 시점 창을 이벤트에 기록합니다.
- 이벤트 sidecar(CNN 입력 창, `p_fall`, 모델 해시)를 저장하고 처리 fps를 노출합니다.
- [owner] 보유한 obs 로그에 라벨을 달고, 배포 카메라로 연출 낙상과 하드 ADL을 녹화합니다.

### Day 3: 리팩토링과 폴더 정리 ✅

판정 로직은 그대로 두고(golden trace로 이벤트 동일성 확인), 이후 모델 작업을 위한 구조를 만들었습니다.

- [x] 안전망: Level1 이벤트 golden trace 테스트, 런타임 모듈 import 스모크 테스트
- [x] 런타임/오프라인 분리: `app/`(Jetson), `training/`(학습), `app/eval/`(평가), `scripts/jetson/`(장치 도구), `eval/`(평가 입력)
- [x] `app/core`는 판단 로직만: controller -> `app/`, streamer -> `app/api/`, clip_recorder/camera/replay -> `app/media/`, PresenceEngine 분리
- [x] 중복 제거: 키포인트 정규화 3곳 -> `app/ai_classifier/features.py`(비트 단위 동일성 테스트), JPEG 인코딩 4곳 -> `app/media/jpeg.py`
- [x] 죽은 코드 삭제(`app/auth`, `_qr_deblur_helper`, `MetricsCollector`, 미사용 메서드/상태/import), 환경변수 읽기를 `config.py`로 일원화
- [x] 줄바꿈 LF 고정, `.dockerignore`, `requirements-train.txt`

## 다음 일정: 1D CNN 고도화 (데이터 -> 구조 -> 보정 순서, 효과를 단계별로 분리 측정)

현재 `Fall1DCNN` 진단: 파라미터 204,290개 중 60%가 flatten 뒤 FC(122,944개)에 몰려 있고,
수용영역은 10프레임(Jetson 기준 약 0.7초)이며 입력 길이가 30프레임으로 고정되어 있습니다.
모델이 낙상 동작이 아닌 "누워서 정지" 구간을 보는 문제를 먼저 고쳐야 구조 개선이 의미를 가집니다.

### Day 4: 학습과 운영 입력 통일 ✅ (모델은 그대로)

- [x] `features.py` v2 (92채널): 타임스탬프 기준 15Hz 리샘플링(0.2초 이하 공백만 보간, 나머지는 마스크),
  등방 픽셀 + 몸통 길이 스케일 정규화, 초당 속도, 전역 골반 궤적, 몸통 각도, bbox 종횡비, conf 마스크, 좌우 반전
- [x] 클립 키포인트 형식(`training/datasets/clipfmt.py`)과 추출기(`extract_keypoints.py`):
  운영과 같은 대상 선택(`app/engine/target_select.py`로 분리), 이어서 실행, 영상 단위 실패 격리
- [x] 통합 manifest(`manifest.py`, `build_manifest.py`): OmniFall 구간 라벨 + URFD 프레임 라벨(`urfd_labels.py`) +
  Figshare 폴더 라벨, 2,726개 영상, 다시점/중복 영상 event_group 병합, URFD RGB 영역 crop
- [x] `kp_buffer`/sidecar에 frame_wh 기록, 가중치 `<weights>.meta.json` 검증(불일치 시 fail-open), 학습 시 meta 자동 기록
- [x] 전체 키포인트 추출 (RTX 4050 Laptop, 52만 프레임, 약 80분, 실패 0건): `python -m training.datasets.extract_keypoints --manifest dataset/_manifest/manifest_v1.jsonl --device 0`

### 단계별 개선 실험 (progression_v1) ✅

Day 5~6 항목 중 분할, 창 생성, 모델(Fall1DCNN-GAP, FallTCN, 고전 ML), 학습 하네스를 먼저 만들어
"규칙만 → 1D CNN + URFD(원래 방식) → 데이터 추가 → 라벨/학습-운영 정렬 → 사건 구간 v2 + 모델 변경 → 임계값 조정"을
같은 5-fold, 같은 이벤트 단위 프로토콜로 측정했습니다. 결과와 주의점은 [MODEL_STUDY.md](MODEL_STUDY.md).

- 원래 방식(1D CNN + URFD)은 규칙만보다 나빴고(balanced 0.635 → 0.434), 데이터 추가가 가장 큰 개선(+0.248)
- 사건 구간 v2(E)는 ADL 오탐을 0.174 → 0.05~0.065로 줄이는 대신 recall이 감소, balanced 차이는 fold 잡음 안
- 다음 개선 후보: LEVEL1 시점의 최신 창(또는 진입 이후 최대 p_fall)으로 판정, onset 오프셋을 fold별로 추정,
  시간 기준 탐지 지표를 실행기에 추가, 규칙 엔진 recall 상한(0.608) 개선

### Day 5: 라벨, 창, 분할과 데이터 분석 (CPU)

- `training/windows.py`: 낙상 onset 기준 양성 창, ADL/onset 이전 음성 창, 룰 트리거 창 하드 네거티브 태깅
- `training/splits.py`: StratifiedGroupKFold(5, event_group), fold 배정 저장과 누수 테스트
- Data card / EDA: fps 분포, 관절 결측률, 낙상 vs 하드 ADL 하강 속도 분포
- 고전 ML 비교군: 손으로 만든 시계열 특징 + LogReg/GBDT, 특징 중요도
- Day 4 리뷰에서 넘어온 주의점: Figshare 클립은 약 1.9초라 4초 창에 못 미침(짧은 창 또는 제외 결정 필요),
  추출기는 운영의 품질 게이트(quality < 0.15, 대상 전환 후 1초 유예)를 적용하지 않음(Day 7 런타임 정렬 때 맞춤),
  Le2i subject 값은 OmniFall 그대로라 교차 피험자 분할 전 확인 필요,
  MCFD 낙상 구간 26/184개는 관절 가시율 30% 미만(특히 4번 카메라)이라 가시율 기준 필터 필요

### Day 6: 모델 zoo와 학습 하네스

- `app/ai_classifier/models/`: `Fall1DCNN`(v1 유지), `Fall1DCNN-GAP`(81k), `FallTCN-40`(43k, dilation 1/2/4/8,
  수용영역 61스텝, attention pooling, 프레임 보조 head), `GRU-64`
- `training/augment.py`: 속도/fps 교란, 앵커 jitter, 좌우 flip(관절 swap), 스케일/종횡비, 관절 dropout
- `training/train_cv.py`: fold x seed, 이벤트 가중 CE, OOF 기반 선택, `runs/train/<id>/` 기록
- [owner, GPU] ablation R0~R8 실행 (분할 수정 -> 라벨/창 -> 특징 -> 헤드 -> TCN -> 보조 head)

### Day 7: 보정과 런타임 정렬

- `training/calibrate.py`: temperature scaling, OOF recall 제약 억제 임계값, reliability diagram -> `meta.json`
- v2 모델은 ABNORMAL 진입 사건 구간을 분류 (v1 경로와 공존, shadow로 먼저 운영)
- `KnownIssuesTest` 2건 해소 (CNN 창이 낙상 onset 미포함, LEVEL1 이후 enter/exit 반복), `replay_eval`에 v2 모드 추가
- 타깃 연속성: 후보가 1명이면 switch 판정 금지, ABNORMAL/억제 감시 중 스위치 리셋 금지, lost 중 승격 평가

### Day 8: 강건성, 배포 증거, 피드백 루프

- 강건성: fps 10/20Hz, 관절 dropout, EMA vs 원시 입력, 카메라 시점
- `training/export_onnx.py`와 PyTorch-ORT parity 테스트, Jetson 지연(ORT CPU / TensorRT FP16)
- 클립 검수 큐, `retrain_from_runs`의 무검수 라벨 0 제거 (승격된 실제 낙상도 `not_fall`에 저장되기 때문)
- [owner, GPU/Jetson] 실험 실행과 지연 측정

### Day 9: 일반화와 포트폴리오 정리

- Le2i 어댑터와 cross-dataset 평가 (URFD+커스텀 학습 -> Le2i 테스트)
- `docs/MODEL_STUDY.md`(진단, ablation 표, 신뢰구간, attention 그림), `DATA_CARD.md`, `RESEARCH_SKELETON_MODELS.md`
  (TCN, InceptionTime, MiniRocket, ST-GCN, Transformer 비교와 채택 근거)
- (선택) 사전학습 ST-GCN++ 전이 학습 기술 검증

주의:

- 라벨, 특징, 구조를 한 실험에서 동시에 바꾸지 않습니다.
- 낙상 이벤트가 약 30개라 1개 차이가 recall 약 17%p입니다. 신뢰구간 안의 차이는 "구분 불가"로 씁니다.
- 합성 시나리오 결과는 제어 흐름 회귀 테스트일 뿐, 정확도 지표로 쓰지 않습니다.
