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
- 낙상 시 `runs/clips/`에 파일 생성 여부

## 저장 파일

런타임 중 생성될 수 있는 파일:

- `runs/device_state.json`
- `runs/*.jsonl`
- `runs/clips/fall/*.mp4`
- `runs/clips/not_fall/*.mp4`

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
