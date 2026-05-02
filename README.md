# Fall Detection Edge App

Jetson Orin Nano 환경에서 동작하는 실시간 낙상 감지 엣지 애플리케이션입니다.  
카메라 영상에서 사람의 자세를 추출한 뒤, 1차 규칙 기반 판단과 2차 딥러닝 분류를 결합해 낙상 여부를 판정합니다.

이 저장소는 단순한 모델 추론 데모가 아니라, 실제 현장 장치 관점의 흐름을 포함합니다.

- 장치 미등록 시 QR 페어링 모드로 시작
- 등록 완료 후 실시간 감시 모드로 자동 전환
- YOLO Pose 기반 키포인트 추출
- 좌표 변화 기반 1차 낙상 의심 탐지
- 1D CNN 기반 2차 낙상/일상행동 재검증
- 사고 전후 영상 클립 저장
- 백엔드 서버 및 하위 장치(Raspberry Pi) 이벤트 전송
- MJPEG 스트리밍과 WebSocket 프레임 릴레이 지원

## 프로젝트 목적

이 프로젝트의 목표는 사람의 낙상을 가능한 한 빠르게 감지하면서도, 단순 자세 변화나 일상 동작을 낙상으로 오인하는 비율을 줄이는 것입니다.

이를 위해 시스템은 두 단계로 동작합니다.

1. 빠른 1차 판단
좌표, 자세 비율, 중심 이동, 어깨 하강, 저자세 지속 여부를 이용해 낙상 의심 상황을 탐지합니다.

2. 느리지만 더 신중한 2차 판단
최근 30프레임의 키포인트 시퀀스를 1D CNN 분류기에 넣어, 규칙 기반 탐지 결과를 다시 검증합니다.

즉, 이 프로젝트는 "실시간성"과 "오탐 억제" 사이의 균형을 맞추는 엣지 비전 시스템입니다.

## 시스템 개요

전체 흐름은 아래와 같습니다.

```text
카메라 입력
  -> 실행 모드 선택
     -> QRMode: 장치 등록용 QR 스캔
     -> LiveMode: 실시간 낙상 감시
  -> YOLO Pose 추론
  -> Observation 표준화
  -> EMA 스무딩
  -> Level1Engine 규칙 기반 판단
  -> FallClassifier 2차 검증
  -> 이벤트 전송 / 클립 저장 / 스트리밍
```

### 실행 모드

이 애플리케이션은 시작 시 등록 상태를 확인하고 자동으로 모드를 선택합니다.

- `QRMode`
  장치가 아직 서버에 등록되지 않은 경우 실행됩니다. 카메라 프레임에서 QR 코드를 읽고, 서버에 페어링 요청을 보냅니다.
- `LiveMode`
  장치가 이미 등록된 경우 실행됩니다. YOLO Pose 추론, 재실 감지, 낙상 판단, 클립 저장, 이벤트 전송을 수행합니다.

QR 등록이 완료되면 프로세스를 재시작하지 않아도 `QRMode -> LiveMode`로 자동 전환됩니다.

## 핵심 아이디어

### 1. 1차 낙상 판단: 규칙 기반 엔진

`app/core/level1.py`의 `Level1Engine`이 핵심입니다.

이 엔진은 한 프레임만 보고 판단하지 않고, 상태 전이 방식으로 동작합니다.

- `NORMAL`
  정상 상태
- `ABNORMAL`
  낙상 의심 상태
- `LEVEL1`
  낙상 확정 상태

판단에 사용하는 주요 신호는 다음과 같습니다.

- 사람 바운딩 박스 중심의 급격한 하강
- 세로/가로 비율 변화
- 어깨 키포인트의 빠른 하강
- 저자세가 일정 시간 이상 유지되는지 여부
- 낙상 이후 정지 상태가 지속되는지 여부
- 대상 스위칭 발생 여부

이 구조 덕분에 단순히 앉거나 허리를 굽히는 동작과 실제 낙상 가능 상황을 구분하려고 시도합니다.

### 2. 2차 낙상 판단: 딥러닝 재검증

규칙 기반 엔진이 `LEVEL1`로 가기 직전, 최근 30프레임의 키포인트 시퀀스를 1D CNN 분류기에 전달합니다.

- 입력: 17개 키포인트 x/y 좌표 시퀀스
- 정규화: 골반 중심 기준 상대좌표화
- 출력: `Fall` 또는 `ADL(일상행동)`

AI 분류기가 `ADL`로 판정하면 규칙 기반 탐지를 억제하고 `ai_suppressed` 이벤트로 기록합니다.  
즉, 이 저장소의 핵심 설계는 "좌표 기반 1차 탐지 + 딥러닝 2차 필터"입니다.

### 3. 영상 클립 저장

낙상 의심 시점 전후 구간을 저장하기 위해 링버퍼 기반 `ClipRecorder`를 사용합니다.

- 평소에는 최근 프레임을 JPEG로 압축해서 메모리에 유지
- 사고 확정 시 `pre/post` 구간을 잘라 MP4로 저장
- AI에 의해 반려된 사례도 `not_fall` 데이터로 저장 가능

이 구조는 사후 확인, 액티브 러닝, 데이터셋 축적에 유리합니다.

## 주요 컴포넌트

### 앱 진입점

- `app/main.py`
  FastAPI 앱 생성, 컨트롤러 시작/종료, API 라우트 정의

### 실행 제어

- `app/core/controller.py`
  애플리케이션의 실제 메인 루프입니다.
  카메라 열기, 현재 모드 선택, 프레임 버퍼 갱신, 리플레이, JSONL 레코딩을 담당합니다.

### 실행 모드

- `app/modes/qr_mode.py`
  QR 인식 및 서버 페어링 담당
- `app/modes/live_mode.py`
  실시간 감시 모드 담당
- `app/modes/base_mode.py`
  모든 모드의 공통 인터페이스

### 추론 파이프라인

- `app/engine/live.py`
  카메라 프레임 읽기, YOLO Pose 추론, 대상 선택, Observation 생성, 시각화
- `app/engine/smoothing.py`
  키포인트 EMA 스무딩

### 낙상 판단

- `app/core/level1.py`
  규칙 기반 낙상 판단 엔진
- `app/ai_classifier/model.py`
  1D CNN 모델 정의
- `app/ai_classifier/inference.py`
  2차 분류기 로딩 및 추론

### 데이터 구조

- `app/core/obs_schema.py`
  YOLO 결과를 공통 Observation 구조로 변환

### 클립 저장

- `app/core/clip_recorder.py`
  사고 전후 프레임 저장 및 MP4 생성

### 장치 상태 / 외부 통신

- `app/state/device_state.py`
  등록 상태, 토큰, 갱신 정보 저장
- `app/api/server_client.py`
  페어링, 이벤트 보고, 토큰 갱신, 업로드 API 호출
- `app/core/streamer.py`
  원본 프레임을 WebSocket으로 백엔드에 릴레이

### 유틸리티

- `app/utils/camera.py`
  OS별 카메라 열기, 해상도 설정, 오토포커스 제어
- `app/utils/replay.py`
  저장된 Observation JSONL 기반 리플레이

## 저장소 구조

```text
.
├── app/
│   ├── ai_classifier/      # 2차 검증 모델, 학습/추론 코드
│   ├── api/                # 서버/RPI 통신 클라이언트
│   ├── core/               # 컨트롤러, 낙상 엔진, 클립 저장
│   ├── engine/             # YOLO 추론 파이프라인, 스무딩
│   ├── modes/              # QR 모드, Live 모드
│   ├── state/              # 장치 등록 상태 저장
│   ├── utils/              # 카메라, 리플레이, 메트릭 유틸
│   ├── config.py           # 환경 변수 기반 설정
│   └── main.py             # FastAPI 엔트리포인트
├── docs/                   # 상세 기술 문서
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── benchmark_models.py     # 모델 벤치마크 도구
├── export_tensorrt.py      # TensorRT 변환 스크립트
├── test_confidence.py      # 낙상 신뢰도 로직 테스트
└── test_token_refresh.py   # 토큰 갱신 로직 테스트
```

## 실행 환경

주 대상 환경은 Jetson Orin Nano입니다.

### 가정하는 런타임

- Ubuntu/Linux 기반 Jetson 환경
- NVIDIA 런타임 사용 가능
- USB 또는 CSI 카메라 연결
- PyTorch + CUDA 동작 가능

### 사용 기술

- Python
- FastAPI
- OpenCV
- Ultralytics YOLO Pose
- PyTorch
- WebSocket
- Docker / Docker Compose

## 빠른 시작

### 1. Docker Compose 실행

```bash
docker-compose up -d --build
```

기본 `docker-compose.yml`은 다음을 전제로 합니다.

- 컨테이너 이름: `jetson-edge`
- 포트: `8000`
- 카메라 디바이스: `/dev/video0`
- `runs/` 디렉터리 호스트-컨테이너 마운트
- NVIDIA 런타임 사용

### 2. 서버 상태 확인

```bash
curl http://localhost:8000/health
```

정상 동작 시 `processing_mode`, `is_registered`, `replay_running` 등의 상태를 확인할 수 있습니다.

### 3. 스트림 확인

이 프로젝트는 현재 두 개의 MJPEG 스트림 엔드포인트를 제공합니다.

- 오버레이 포함 분석 화면  
  `GET /api/iot/device/falls/stream_overlay`
- 오버레이 없는 원본 화면  
  `GET /api/iot/device/falls/stream`

예시:

```bash
curl http://localhost:8000/api/iot/device/falls/stream_overlay -o overlay.mjpeg
curl http://localhost:8000/api/iot/device/falls/stream -o raw.mjpeg
```

## API 요약

### 상태 조회

- `GET /health`
  앱 전체 상태
- `GET /level1/status`
  낙상 관련 최근 상태와 클립 레코더 상태
- `GET /pose`
  가장 최근 Observation 반환

### 스트리밍

- `GET /api/iot/device/falls/stream_overlay`
- `GET /api/iot/device/falls/stream`

### 레코딩

- `POST /record/start`
  Observation JSONL 기록 시작
- `POST /record/stop`
  Observation JSONL 기록 종료

### 리플레이

- `POST /replay/start?path=<absolute_path>&fps=15.0`
- `POST /replay/stop`

### 모드 관련

- `POST /mode/live`
  수동 전환용이 아니라 현재는 자동 모드 동작 안내용
- `POST /mode/replay`
  리플레이 진입 안내용

## Observation 데이터 구조

실시간 추론 결과는 내부적으로 Observation이라는 공통 구조로 정리됩니다.

```json
{
  "schema_version": "1.0",
  "ts": 1710000000.0,
  "frame_index": 123,
  "source_id": "cam0",
  "target_switched": false,
  "meta": {
    "conf_thres": 0.5
  },
  "tracks": [
    {
      "track_id": 0,
      "has_person": true,
      "bbox": [0.1, 0.2, 0.4, 0.8],
      "conf": 0.91,
      "keypoints_raw": [],
      "keypoints_smooth": [],
      "keypoints": [],
      "quality_score": 0.74,
      "frame_shape": [1080, 1920]
    }
  ]
}
```

이 구조를 기준으로 낙상 엔진, 리플레이, 기록, 후처리 로직이 연결됩니다.

## 자동 모드 전환 흐름

```text
앱 시작
  -> DeviceState 확인
     -> 등록 안 됨: QRMode
     -> 등록됨: LiveMode

QRMode
  -> 카메라 프레임 캡처
  -> QR 검출
  -> 서버 페어링 성공
  -> DeviceState 저장
  -> LiveMode로 자동 전환

LiveMode
  -> YOLO Pose 추론
  -> Observation 생성
  -> EMA 스무딩
  -> PresenceEngine
  -> Level1Engine
  -> AI 재검증
  -> 이벤트 전송 / 클립 저장 / 스트리밍
```

## 주요 설정값

설정은 대부분 `app/config.py`에서 환경 변수로 읽습니다.

### 카메라 / 영상

- `CAM_INDEX`
- `FRAME_W`
- `FRAME_H`
- `JPEG_QUALITY`
- `DEFAULT_CONF`

### 모델 / 연산

- `MODEL_PATH`
- `DETERMINISTIC`
- `USE_HALF`

### 장치 / 서버

- `DEVICE_ID`
- `DEVICE_NAME`
- `LOCATION_ID`
- `SERVER_URL`
- `WS_SERVER_URL`
- `RPI_URL`

### 저장 경로

- `RUNS_DIR`
- `CLIP_DIR`

### 사고 클립

- `CLIP_FPS`
- `CLIP_PRE_SEC`
- `CLIP_POST_SEC`
- `CLIP_RESIZE_WIDTH`
- `CLIP_RESIZE_HEIGHT`

### 낙상 판단 / 스무딩

- `ABNORMAL_TIMEOUT_S`
- `KP_EMA_ENABLE`
- `KP_EMA_ALPHA`
- `KP_MIN_CONF`

실제 알고리즘 세부 파라미터는 `Level1Params`, `PresenceParams`, 스무딩 모듈 내부에도 정의되어 있습니다.

## 현재 구현된 기능

- 자동 QR/Live 모드 전환
- 실시간 카메라 추론
- 단일 대상 선택 및 추적
- 키포인트 EMA 스무딩
- 재실 감지 이벤트
- 낙상 의심/확정 상태 전이
- AI 기반 2차 오탐 억제
- 사고 전후 영상 저장
- 서버 이벤트 전송
- 토큰 만료 시 자동 갱신
- WebSocket 프레임 릴레이
- Observation JSONL 기록 및 리플레이

## 이 저장소를 포트폴리오 관점에서 볼 때의 포인트

이 프로젝트는 다음 이유로 단순 모델 추론 코드보다 설명 가치가 있습니다.

- 추론 코드만 있는 것이 아니라 장치 상태와 운영 모드가 분리되어 있음
- 낙상 감지를 상태 머신으로 설계함
- 규칙 기반과 딥러닝 기반을 결합한 하이브리드 구조를 가짐
- 실제 운영을 고려한 클립 저장, 토큰 갱신, 이벤트 전송이 포함됨
- 액티브 러닝용 `not_fall` 사례 축적 흐름이 있음

즉, "모델을 돌렸다"가 아니라 "엣지 장치에서 운영되는 낙상 감지 시스템을 설계하고 구현했다"는 점이 핵심입니다.

## 함께 보면 좋은 문서

- [docs/README.md](docs/README.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/FALL_DETECTION_CORE.md](docs/FALL_DETECTION_CORE.md)
- [docs/OPERATIONS.md](docs/OPERATIONS.md)
- [docs/TUNING_GUIDE.md](docs/TUNING_GUIDE.md)

## 한계와 참고 사항

- 현재 구현은 단일 주요 대상 추적에 가깝습니다.
- 멀티인원 장면에서 완전한 ID 추적 시스템은 아닙니다.
- 실제 서버 주소, 디바이스 ID, 하위 장치 주소 등은 환경에 맞게 조정해야 합니다.
- Jetson 실환경 최적화는 카메라, 모델 크기, TensorRT 적용 여부에 따라 달라집니다.
- 저장소에는 학습 데이터와 가중치, 실행 산출물이 함께 존재할 수 있으므로 공개 전 별도 정리가 필요합니다.

## 다음 정리 권장 사항

이 README 이후 공개 저장소 품질을 더 높이려면 아래 순서가 적절합니다.

1. `.gitignore` 정리
2. `runs/`, 영상 클립, 대용량 `npy/pth` 파일 정리
3. `docs/` 문서와 실제 코드 값 동기화
4. `requirements.txt` 보강
5. `.env.example` 추가

---

이 프로젝트는 Jetson 기반 실시간 비전 시스템, 낙상 감지 알고리즘, 엣지-백엔드 연동, 운영형 애플리케이션 구조를 함께 보여주기 위한 코드베이스입니다.
