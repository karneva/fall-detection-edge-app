# Operations

## 실행 방식

이 프로젝트는 FastAPI 서버로 실행되며, 시작 시 백그라운드 처리 루프와 WebSocket 스트리머가 함께 구동됩니다.

기본 실행 방식:

```bash
cp .env.example .env
```

```bash
docker-compose up -d --build
```

기본 포트:

- `8000`

의존성 기준:

- Python 패키지: `requirements.txt`
- PyTorch/CUDA: Jetson 베이스 이미지에서 제공
- QR 인식용 시스템 라이브러리: `libzbar0`

환경값 주입 기준:

- `docker-compose.yml`은 `.env` 또는 셸 환경변수를 읽어 앱 설정값을 전달합니다.
- 공개 저장소에서는 `.env.example`을 템플릿으로 사용하면 됩니다.

## 시작 시 동작

애플리케이션 시작 후 순서는 다음과 같습니다.

1. `AppController` 시작
2. 카메라 열기
3. `DeviceState` 확인
4. 미등록이면 `QRMode`, 등록되어 있으면 `LiveMode`
5. 프레임 처리 루프 시작
6. `WebSocketStreamer`가 백엔드 WebSocket 연결 시도

## 주요 엔드포인트

### 상태 조회

- `GET /health`
  - 전체 상태
  - 현재 처리 모드
  - 등록 여부
  - 리플레이 실행 여부

- `GET /level1/status`
  - 최근 낙상 관련 상태
  - 마지막 `level1` 이벤트
  - 클립 레코더 상태

- `GET /pose`
  - 최신 Observation 반환

### MJPEG 스트리밍

- `GET /api/iot/device/falls/stream_overlay`
  - 스켈레톤/박스가 표시된 오버레이 화면

- `GET /api/iot/device/falls/stream`
  - 원본 프레임

### 레코딩

- `POST /record/start`
  - Observation JSONL 기록 시작

- `POST /record/stop`
  - Observation JSONL 기록 종료

### 리플레이

- `POST /replay/start?path=<absolute_path>&fps=15.0`
  - 기록된 JSONL 기반 리플레이 시작

- `POST /replay/stop`
  - 리플레이 종료

## 모드별 운영 포인트

### QRMode

확인할 것:

- 카메라 초점
- QR 인식률
- 서버 페어링 응답
- `runs/device_state.json` 생성 여부

문제가 생기면 주로 아래를 봅니다.

- `SERVER_URL`
- `DEVICE_ID`
- `pyzbar` 설치 여부
- 카메라 접근 가능 여부

### LiveMode

확인할 것:

- YOLO 모델 로드 성공
- `/pose` 응답 생성 여부
- `/health`의 `processing_mode=live`
- 낙상 시 `runs/clips/`, `runs/events/`에 파일 생성 여부
- `/health`의 `ai_classifier.enabled`, `processing_fps`

### 앞단 기본값 (Day 8부터 켜짐)

| 변수 | 기본값 | 운영에서 볼 점 |
| --- | --- | --- |
| `ROTATE_FALLBACK` | `1` | 사람을 놓친 프레임에서 추론이 2회 늘어남. 사람을 마지막으로 본 뒤 5초 안에만 동작하므로 빈 방에서는 부하 없음. 사람이 자주 끊기는 장면에서 `processing_fps`가 떨어지면 먼저 확인 |
| `LOST_GRACE_S` | `0.5` | 0.5초보다 짧은 미검출에는 판정 상태를 유지 |
| `TARGET_SWITCH_POLICY` | `multi` | 후보가 1명이면 대상 전환(엔진 리셋)이 일어나지 않음. 로그 `[SWITCH DETECTED]`는 여러 사람이 보일 때만 나와야 정상 |

- 오탐이 늘었다는 현장 보고가 있으면 `ROTATE_FALLBACK=0`부터 끄고 비교합니다 (근거와 교환은 [TUNING_GUIDE](TUNING_GUIDE.md#1-1-앞단-개선-r1-r2-회전-폴백)).
- 세 값을 모두 끄면 Day 7 이전 동작입니다. 바꾼 뒤에는 컨테이너를 다시 시작해야 반영됩니다.
- JSONL 리플레이는 기록 당시의 `target_switched`를 그대로 씁니다 (대상 전환 정책은 실시간 처리에서만 적용).
- LEVEL1 알람 뒤 사람이 계속 누워 있으면 `abnormal_enter`/`abnormal_exit` 이벤트가 몇 초마다 반복 기록됩니다 (알림은 없음).
  `POST_ALARM_LATCH=1`로 알람 뒤 래치를 켜면 반복이 없어지지만, 앉았다 다시 넘어지는 경우 등 두 번째 알람을 놓칠 수 있어
  기본으로 끕니다 ([TUNING_GUIDE](TUNING_GUIDE.md)). 래치를 켰을 때:
  일어선 자세(넘어지기 전 자세 기준)가 약 0.4초 이상 보이면 진입 차단이 풀려, 그 뒤 다시 넘어지면 빠르게든 천천히든
  (합성 시험: 0.3초 서 있다가 5초에 걸쳐 무너짐까지) 다시 알람이 나갑니다. 누운 채 bbox가 한두 프레임 튀는 것은 일어섬으로 보지 않습니다.
  알람 뒤 쿨다운(`level1_cooldown_s`, 25초) 동안은 원래대로 판정하지 않습니다.

## 저장 파일

런타임 중 생성될 수 있는 파일:

- `runs/device_state.json`
- `runs/*.jsonl`
- `runs/clips/fall/*.mp4`
- `runs/clips/not_fall/*.mp4` (AI가 억제한 사건, 검수 전에는 라벨 미확정)
- `runs/events/*.json` (낙상/억제 이벤트 sidecar: 판정 근거 키포인트, AI 입력, 파라미터. `SAVE_EVENT_SIDECAR=0`으로 끔)
- `runs/eval/*.json` (`app.eval.replay_eval` 결과)

공개 저장소에서는 이 파일들을 커밋하지 않는 것이 맞습니다.

## 토큰과 장치 상태

장치 등록 성공 후 다음 정보가 로컬에 저장됩니다.

- `device_id`
- `access_token`
- `refresh_token`
- `group_id`
- `serial_number`
- `token_expiry`

운영 중 액세스 토큰이 만료되면 `DeviceState.ensure_valid_token()`이 리프레시를 시도하고, 성공하면 RPI에도 갱신 토큰을 전파합니다.

## 이벤트 전송

### RPI로 보내는 이벤트

- enter
- exit
- fall_detected

### 백엔드 서버로 보내는 이벤트

- QR 페어링
- 낙상 감지 보고
- 영상 업로드 완료 보고

## WebSocket 스트리밍

`WebSocketStreamer`는 원본 MJPEG 프레임을 백엔드 서버로 전송합니다.

특징:

- 연결 실패 시 재시도
- 시작 시 `REGISTER_DEVICE` 메시지 전송
- 원본 프레임 기준 스트리밍

## 리플레이 모드

리플레이는 카메라 대신 저장된 Observation JSONL을 재생합니다.

용도:

- 디버깅
- 알고리즘 검증
- 튜닝 전후 비교

제약:

- 원본 영상이 아니라 Observation 기반 렌더링 화면

## 운영 시 추천 체크 순서

1. `/health` 확인
2. `/pose` 확인
3. 오버레이 스트림 확인
4. 서버/RPI 이벤트 로그 확인
5. `runs/clips/` 생성 확인

## 자주 보는 장애 원인

- 카메라 장치 권한 문제
- YOLO 모델 경로 오류
- 서버 주소 또는 네트워크 문제
- QR 라이브러리 미설치
- 대용량 `runs/` 누적으로 인한 디스크 사용량 증가
