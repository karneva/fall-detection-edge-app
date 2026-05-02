# Architecture

## 개요

이 프로젝트는 Jetson Orin Nano에서 동작하는 실시간 낙상 감지 애플리케이션입니다.  
카메라 입력을 받아 포즈를 추론하고, 규칙 기반 1차 판단과 딥러닝 기반 2차 판단을 조합해 낙상 여부를 결정합니다.

애플리케이션은 FastAPI 서버 형태로 실행되지만, 핵심은 HTTP 서버가 아니라 백그라운드 처리 루프입니다.

## 상위 구조

```text
FastAPI app
  -> AppController
     -> CameraManager
     -> Mode selection
        -> QRMode
        -> LiveMode
     -> Frame buffers
     -> Replay / Record
  -> WebSocketStreamer
```

## 실행 모드

### QRMode

목적:

- 등록되지 않은 장치를 서버에 페어링
- QR 코드 인식
- 등록 성공 후 `DeviceState` 갱신

구현 위치:

- `app/modes/qr_mode.py`

특징:

- OpenCV 프레임을 읽음
- CLAHE, 샤프닝, 디블러링 등 전처리를 적용
- `pyzbar` 기반 QR 검출
- 등록 성공 시 다음 루프에서 `LiveMode`로 자동 전환

### LiveMode

목적:

- 실시간 사람 감지 및 낙상 감시
- 재실 이벤트 전송
- 사고 클립 저장
- 서버 업로드

구현 위치:

- `app/modes/live_mode.py`

특징:

- `LivePipeline`으로 YOLO Pose 추론 수행
- `PresenceEngine`으로 enter/exit 이벤트 판단
- `Level1Engine`으로 낙상 판단
- `ClipRecorder`로 사고 전후 영상 보존

## 처리 흐름

```text
Camera frame
  -> YOLO Pose inference
  -> best-person selection
  -> Observation build
  -> EMA smoothing
  -> PresenceEngine
  -> Level1Engine
  -> AI verification
  -> event / clip / stream
```

## 주요 모듈 역할

### `app/main.py`

- FastAPI 엔트리포인트
- `AppController`, `WebSocketStreamer` 시작/종료
- 상태 조회, 스트리밍, 레코딩, 리플레이 API 정의

### `app/core/controller.py`

- 실제 메인 제어 루프
- 현재 모드 선택
- 카메라 상태 유지
- 최신 오버레이/원본 JPEG 버퍼 관리
- Observation JSONL 기록
- 리플레이 시작/종료

### `app/engine/live.py`

- 카메라 프레임 읽기
- YOLO Pose 추론
- 대상 1명 선택
- Observation 생성
- 시각화 이미지 생성

이 모듈은 멀티 타겟 추적 시스템이 아니라, 가장 중요한 1명의 대상에 집중하는 구조입니다.

### `app/core/obs_schema.py`

YOLO 결과를 아래 공통 구조로 정규화합니다.

```json
{
  "schema_version": "1.0",
  "ts": 0.0,
  "frame_index": 0,
  "source_id": "cam0",
  "target_switched": false,
  "tracks": [
    {
      "track_id": 0,
      "has_person": true,
      "bbox": [0.0, 0.0, 1.0, 1.0],
      "conf": 0.0,
      "keypoints_raw": [],
      "keypoints_smooth": [],
      "keypoints": [],
      "quality_score": 0.0,
      "frame_shape": [1080, 1920]
    }
  ]
}
```

이 구조를 기준으로 감지, 리플레이, 저장, 후처리가 연결됩니다.

### `app/engine/smoothing.py`

- 키포인트 EMA 스무딩
- 관절별 알파 조정
- 신뢰도 기반 홀드
- 급격한 점프 이동 게이팅
- 대상 변경 또는 해상도 변경 시 상태 초기화

### `app/core/level1.py`

- 규칙 기반 낙상 판단 상태 머신
- `NORMAL -> ABNORMAL -> LEVEL1`
- 재실 감지 엔진 포함
- 신뢰도 점수 계산
- 2차 AI 검증 연결

### `app/ai_classifier/*`

- `model.py`: 1D CNN 구조
- `inference.py`: 모델 로딩, 시퀀스 정규화, 추론
- `train*.py`: 학습 파이프라인

### `app/core/clip_recorder.py`

- 최근 프레임을 JPEG 링버퍼로 유지
- 낙상 발생 시 전후 구간 MP4 저장
- `fall/`, `not_fall/` 디렉터리 분리 저장

### `app/state/device_state.py`

- 장치 등록 정보 저장
- 액세스/리프레시 토큰 보관
- 토큰 만료 시 자동 갱신

### `app/api/server_client.py`

- QR 페어링 API 호출
- RPI 이벤트 전송
- 서버 낙상 이벤트 보고
- Presigned URL 기반 영상 업로드

## 백그라운드 스레드 구조

### 앱 처리 스레드

`AppController.start()`가 실행되면 백그라운드 루프가 시작됩니다.

이 루프는:

- 리플레이 실행 중인지 확인
- 모드가 없으면 `QRMode` 또는 `LiveMode` 생성
- 현재 모드의 `step()` 호출
- 최신 프레임 버퍼 갱신
- Observation 기록

### WebSocket 스트리밍 스레드

`WebSocketStreamer`는 원본 MJPEG 프레임을 백엔드 WebSocket 서버로 전송합니다.

용도:

- 원격 모니터링
- 서버 측 연동
- 장치 등록 메시지 전송

## 카메라 제어

`CameraManager`가 플랫폼별 카메라 열기와 설정을 담당합니다.

- Linux/Jetson: V4L2 우선
- Windows: DSHOW/MSMF 우선
- 모드에 따라 해상도 조정
  - QR 모드: `640x480`
  - Live 모드: `1920x1080` 요청

## 저장 구조

런타임 데이터는 주로 `runs/` 아래에 쌓입니다.

- `runs/device_state.json`
- `runs/*.jsonl`
- `runs/clips/fall/*.mp4`
- `runs/clips/not_fall/*.mp4`

공개 저장소 기준으로는 이 디렉터리를 Git 추적 대상에서 제외하는 것이 맞습니다.
