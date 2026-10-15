# Fall Detection Edge App

**Jetson Orin Nano 위에서 카메라 한 대로 사람의 낙상을 실시간 감지하는 엣지 애플리케이션**

![Jetson Orin Nano](https://img.shields.io/badge/Jetson-Orin%20Nano-76B900?logo=nvidia&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1D%20CNN-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv8 Pose](https://img.shields.io/badge/YOLOv8-Pose-111F68)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)

YOLOv8-Pose로 사람의 관절 좌표를 뽑고, **규칙 기반 상태 머신이 빠르게 낙상을 의심**한 뒤
**1D CNN이 키포인트 시계열을 보고 한 번 더 검증**합니다. 확정되면 보호자 장치(Raspberry Pi)와 서버에 알리고 사고 전후 영상을 업로드합니다.

> 목표: 실제 낙상은 놓치지 않으면서(높은 recall), 침대에 눕기·빨리 앉기 같은 일상 동작의 오탐은 줄이기

---

## 한눈에 보기

```mermaid
flowchart LR
    CAM["카메라<br/>1920x1080"] --> YOLO["YOLOv8-Pose<br/>관절 17개 추출"]

    subgraph JETSON["Jetson Orin Nano (app/)"]
        YOLO --> OBS["Observation<br/>표준화 + EMA 스무딩"]
        OBS --> RULE["1차 판단<br/>규칙 기반 상태 머신"]
        RULE -- "낙상 의심이 지속되면" --> AI["2차 검증<br/>1D CNN (p_fall)"]
        OBS --> BUF[("키포인트 버퍼<br/>사건 구간 기록")]
    end

    AI -- "낙상 확정" --> ALARM["알람"]
    AI -. "일상 동작으로 판단<br/>(억제 후 계속 감시)" .-> RULE

    ALARM --> RPI["Raspberry Pi<br/>현장 알림"]
    ALARM --> SERVER["백엔드 서버<br/>이벤트 + 영상 업로드"]
    ALARM --> RUNS[("runs/<br/>클립 · sidecar")]
    BUF --> RUNS
```

| 단계 | 하는 일 | 위치 |
| --- | --- | --- |
| 자세 추출 | YOLOv8-Pose로 가장 확실한 1명의 관절 17개 좌표 추출, 대상 전환 감지 | `app/engine/` |
| 1차 판단 | 하강 속도·자세 비율·어깨 하강·높이 붕괴를 조합해 낙상 의심 → 확정 | `app/core/level1.py` |
| 2차 검증 | 키포인트 30프레임을 1D CNN에 넣어 낙상 확률 `p_fall` 계산 | `app/ai_classifier/` |
| 알림·기록 | RPI/서버 이벤트, 사고 전후 MP4 업로드, 판정 근거(sidecar) 저장 | `app/modes/live_mode.py` |

## 동작 방식

### 1. 장치 시작: QR 페어링 → 실시간 감시

앱은 시작할 때 장치 등록 상태를 보고 모드를 고릅니다. QR로 등록을 마치면 재시작 없이 감시 모드로 넘어갑니다.

```mermaid
flowchart LR
    START(["앱 시작"]) --> CHECK{"장치가<br/>등록되어 있나?"}
    CHECK -- "아니오" --> QR["QRMode<br/>카메라로 QR 인식"]
    QR --> PAIR["서버 페어링<br/>토큰 저장"]
    PAIR --> LIVE
    CHECK -- "예" --> LIVE["LiveMode<br/>실시간 낙상 감시"]
```

### 2. 낙상 판단 상태 머신

한 프레임만 보고 판단하지 않습니다. **의심 → 지속 확인 → AI 검증 → 확정** 순서로 상태가 바뀝니다.

```mermaid
stateDiagram-v2
    state "정상 NORMAL" as NORMAL
    state "낙상 의심 ABNORMAL" as ABNORMAL
    state "AI 검증 (1D CNN)" as AI_CHECK
    state "억제 후 감시" as WATCH
    state "낙상 확정 LEVEL1" as LEVEL1

    [*] --> NORMAL
    NORMAL --> ABNORMAL: ① 낙상 신호
    ABNORMAL --> NORMAL: ② 다시 일어남
    ABNORMAL --> AI_CHECK: ③ 쓰러진 채 정지
    AI_CHECK --> LEVEL1: ④ 낙상
    AI_CHECK --> WATCH: ⑤ 일상 동작
    WATCH --> NORMAL: ② 다시 일어남
    WATCH --> LEVEL1: ⑥ 오래 쓰러져 있음
    LEVEL1 --> NORMAL: ⑦ 쿨다운
```

| 전이 | 조건 (기본값) |
| --- | --- |
| ① 낙상 신호 | 아래 신호 중 2개 이상이 0.35초 지속, 또는 누운 자세가 45프레임 지속 |
| ② 다시 일어남 | 서 있는 자세로 2초 회복 |
| ③ 쓰러진 채 정지 | 10초간 움직임 없음, 또는 10초 경과 + 자세 붕괴 확인 |
| ④ 낙상 | AI의 낙상 확률 `p_fall > 0.4`, 또는 AI가 판단할 수 없음 (fail-open) |
| ⑤ 일상 동작 | `p_fall ≤ 0.4` → 알람을 보류하고 계속 감시 |
| ⑥ 오래 쓰러져 있음 | 보류 후 60초 넘게 쓰러져 있고, 진입 시 급하강이 있었음 → AI와 무관하게 알람 |
| ⑦ 쿨다운 | 알람 후 25초 |

판단에 쓰는 신호:

- **하강 속도**: 사람 박스 중심이 빠르게 내려감
- **자세 비율**: 세로로 길던 박스가 가로로 눕는 형태로 바뀜
- **어깨 하강**: 어깨 관절이 빠르게 내려감
- **높이 붕괴**: 박스 높이가 평소보다 크게 줄어듦
- **정지 지속**: 쓰러진 뒤 움직임이 거의 없음

### 3. AI 2차 검증은 알람을 "조용히" 지우지 못하게 설계

딥러닝 모델이 틀려도 실제 낙상 알람이 사라지지 않도록, 판단이 불확실하면 항상 규칙 기반 결과(알람)를 따릅니다.

```mermaid
flowchart TD
    IN(["규칙 엔진이 낙상 확정 직전"]) --> MODE{"AI_MODE"}
    MODE -- "off" --> ALARM["알람"]
    MODE -- "veto / shadow" --> LOADED{"가중치 로드됨?<br/>추론 성공?<br/>값이 0~1?"}
    LOADED -- "아니오 (fail-open)" --> ALARM
    LOADED -- "예" --> P{"p_fall ≤ 0.4 ?"}
    P -- "아니오, 또는 shadow 모드" --> ALARM
    P -- "예, veto 모드" --> SUP["알람 보류<br/>억제 후 감시"]
    SUP -- "60초 넘게 쓰러져 있으면" --> ALARM
```

- `veto`: AI가 일상 동작이라고 하면 알람을 보류 (기본값)
- `shadow`: AI 판단은 기록만 하고 알람은 규칙대로 (현장 검증용)
- `off`: AI를 쓰지 않음

### 4. 낙상 한 건이 처리되는 과정

```mermaid
sequenceDiagram
    autonumber
    participant C as 카메라 + YOLO
    participant E as 낙상 엔진
    participant A as 1D CNN
    participant L as LiveMode
    participant O as RPI · 서버

    C->>E: 매 프레임 관절 좌표
    Note over E: 급하강 + 자세 변화 감지
    E->>L: abnormal_enter (사건 시작 시각 기록)
    Note over E: 진입 -1.5초 ~ +2.5초<br/>키포인트 구간 저장
    Note over E: 10초간 움직임 없음
    E->>A: 최근 30프레임 키포인트
    A-->>E: p_fall = 0.93
    E->>L: level1 (낙상 확정)
    L->>O: 낙상 이벤트 전송
    Note over L: 사고 전후 12초 MP4 저장<br/>판정 근거 sidecar 저장
    L->>O: 영상 업로드
```

## 빠른 시작

### Docker Compose (Jetson)

```bash
cp .env.example .env          # 서버/RPI 없이 실행하려면 ENABLE_* 값을 0으로
docker compose up -d --build
curl http://localhost:8000/health
```

기본 설정: 컨테이너 `jetson-edge`, 포트 `8000`, 카메라 `/dev/video0`, `runs/` 호스트 마운트, NVIDIA 런타임.
PyTorch/CUDA는 베이스 이미지(`dustynv/pytorch`)가 제공합니다.

### 로컬 실행

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # + 환경에 맞는 torch
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 주요 API

| Method | Endpoint | 설명 |
| --- | --- | --- |
| GET | `/health` | 앱 상태 (모드, 등록 여부, AI 분류기 상태, 처리 FPS) |
| GET | `/level1/status` | 최근 낙상 이벤트, 클립 레코더 상태 |
| GET | `/pose` | 가장 최근 Observation |
| GET | `/api/iot/device/falls/stream_overlay` | 분석 오버레이 MJPEG 스트림 |
| GET | `/api/iot/device/falls/stream` | 원본 MJPEG 스트림 |
| POST | `/record/start`, `/record/stop` | Observation JSONL 녹화 (오프라인 평가용) |
| POST | `/replay/start?path=...&fps=15`, `/replay/stop` | 녹화본 리플레이 |

설정은 모두 환경변수로 바꿀 수 있습니다. 전체 목록은 [`.env.example`](.env.example), 튜닝 방법은 [TUNING_GUIDE](docs/TUNING_GUIDE.md)를 보세요.

## 평가와 테스트

모델이나 파라미터를 바꿀 때마다 **이벤트 단위 지표**(낙상 탐지율, 시간당 오탐, 알람 지연)로 비교하고,
기준보다 나빠지면 채택하지 않습니다.

```mermaid
flowchart LR
    REC["장치에서 녹화<br/>/record/start"] --> LBL["라벨 작성<br/>낙상 시작/끝 시각"]
    LBL --> RE["replay_eval<br/>엔진 오프라인 재생"]
    RE --> CMP{"compare_results<br/>baseline 대비"}
    CMP -- "탐지율 유지 + 오탐 비증가" --> OK["채택"]
    CMP -- "하나라도 악화" --> NO["거부"]
```

```bash
# 녹화 데이터 평가 (규칙만 vs 규칙 + AI)
python -m app.eval.replay_eval --labels eval/labels.jsonl --name field_v1
python -m app.eval.compare_results <baseline.json> runs/eval/field_v1_<sha>.json

# 데이터 없이 엔진 동작 확인 (합성 시나리오)
python -m app.eval.replay_eval --synthetic

# 테스트 (판단 엔진은 표준 라이브러리만으로 동작, numpy/torch 테스트는 없으면 skip)
python -m unittest discover -s tests -t .
```

테스트에는 리팩토링 안전망인 **golden trace**(합성 시나리오 180개의 엔진 이벤트 전체 비교)와
AI 실패 시 알람이 유지되는지 확인하는 **fail-open 테스트**가 포함되어 있습니다.

## 학습 데이터와 재학습

- 공개 데이터셋 **URFD**와 직접 촬영한 낙상/일상 동작 데이터로 1D CNN을 학습했습니다.
- 추가로 확보할 공개 데이터(Le2i, CAUCAFall, GMDCSA24, AI-Hub 낙상 데이터 등)의 우선순위와 라이선스는 [DATASETS](docs/DATASETS.md)에 정리했습니다.
- AI가 일상 동작으로 판단한 사건은 `runs/clips/not_fall/`과 `runs/events/`에 쌓여 재학습 재료가 됩니다.

```bash
pip install -r requirements-train.txt     # + CUDA에 맞는 torch (GPU 학습 머신)
python -m training.dataset_builder        # URFD -> 학습 배열
python -m training.train                  # 1D CNN 학습 -> app/ai_classifier/weights/
python -m training.retrain_from_runs      # 현장 클립으로 액티브 러닝 재학습
```

데이터 원본, 학습 배열, 가중치, TensorRT 엔진은 저장소에 포함하지 않습니다.

## 저장소 구조

```text
.
├── app/                    # Jetson 런타임 (uvicorn app.main:app)
│   ├── main.py             #   FastAPI 엔트리포인트
│   ├── controller.py       #   메인 처리 루프, 모드 전환, 녹화/리플레이
│   ├── config.py           #   환경변수 설정
│   ├── core/               #   판단 로직 (표준 라이브러리만): 낙상 엔진, 재실, 사건 버퍼
│   ├── ai_classifier/      #   1D CNN 모델, 추론(fail-open), 학습/운영 공용 특징
│   ├── engine/             #   YOLO Pose 추론, 키포인트 스무딩
│   ├── media/              #   카메라, 클립 녹화, 리플레이, JPEG
│   ├── modes/              #   QR 페어링 모드, 실시간 감시 모드
│   ├── api/                #   서버/RPI 통신, WebSocket 스트리머
│   ├── state/              #   장치 등록·토큰 상태
│   └── eval/               #   오프라인 평가 도구
├── training/               # 학습·데이터 준비 (GPU 학습 머신)
├── scripts/jetson/         # TensorRT 변환, 벤치마크, IP 보고
├── eval/                   # 평가 baseline, 라벨/manifest 예시
├── tests/                  # 회귀 테스트
└── docs/                   # 상세 문서
```

## 설계 포인트

- **하이브리드 판단**: 빠르고 설명 가능한 규칙 엔진 + 오탐을 줄이는 딥러닝 검증
- **안전 우선 AI 연동**: 가중치 누락·추론 오류·이상값이면 알람 유지(fail-open), 억제된 사건도 계속 감시 후 승격
- **측정 기반 개발**: 이벤트 단위 오프라인 평가와 채택 게이트, 합성 시나리오 회귀 테스트
- **현장 데이터 루프**: 사건마다 판정 근거(키포인트 구간, AI 입력, 파라미터)를 sidecar로 저장해 재학습에 활용
- **운영 고려**: QR 페어링, 토큰 자동 갱신, 사고 영상 업로드, 스트리밍, Docker 배포

## 문서

| 문서 | 내용 |
| --- | --- |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | 모듈 구조, 스레드, 데이터 흐름 |
| [FALL_DETECTION_CORE](docs/FALL_DETECTION_CORE.md) | 낙상 판단 알고리즘, AI 게이트 안전 정책 |
| [EVALUATION_PROTOCOL](docs/EVALUATION_PROTOCOL.md) | 이벤트 단위 평가 절차, 채택 규칙, 평가셋 관리 |
| [TUNING_GUIDE](docs/TUNING_GUIDE.md) | 파라미터와 환경변수 튜닝 |
| [OPERATIONS](docs/OPERATIONS.md) | 실행, API, 저장 파일, 운영 점검 |
| [ROADMAP](docs/ROADMAP.md) | 1D CNN 고도화 계획과 진행 상황 |
| [DATASETS](docs/DATASETS.md) | 추가 학습 데이터 후보, 확보 순서, 라이선스 |

<details>
<summary>Observation 데이터 구조 (프레임마다 엔진에 들어가는 입력)</summary>

```json
{
  "schema_version": "1.0",
  "ts": 1710000000.0,
  "frame_index": 123,
  "source_id": "cam0",
  "target_switched": false,
  "tracks": [
    {
      "track_id": 0,
      "has_person": true,
      "bbox": [0.1, 0.2, 0.4, 0.8],
      "conf": 0.91,
      "keypoints_raw": [{"id": 0, "x": 0.25, "y": 0.22, "conf": 0.88}],
      "keypoints_smooth": [],
      "quality_score": 0.74,
      "frame_shape": [1080, 1920]
    }
  ]
}
```

</details>

## 한계

- 한 장면에서 가장 확실한 한 사람을 추적합니다. 다인원 ID 추적 시스템은 아닙니다.
- 카메라 높이와 각도에 따라 임계값 민감도가 달라질 수 있습니다.
- 현재 1D CNN은 낙상 순간이 아니라 확정 직전의 정지 구간을 봅니다. 개선 계획은 [ROADMAP](docs/ROADMAP.md)에 있습니다.
