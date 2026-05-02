# Tuning Guide

## 목적

이 문서는 현재 코드 기준으로 낙상 감지 민감도와 안정성을 조정할 때 어디를 봐야 하는지 정리한 가이드입니다.

튜닝은 크게 네 영역으로 나뉩니다.

1. YOLO Pose 추론 민감도
2. 키포인트 스무딩
3. 규칙 기반 낙상 판단
4. 재실 감지

## 1. YOLO Pose 추론

관련 설정:

- `DEFAULT_CONF` in `app/config.py`
- `MODEL_PATH`

현재 기본값:

- `DEFAULT_CONF = 0.50`

의미:

- 낮출수록 사람과 관절을 더 쉽게 잡음
- 높일수록 더 확실한 결과만 채택

튜닝 방향:

- 사람은 보이는데 키포인트가 자주 비면 `DEFAULT_CONF`를 낮춤
- 잡음과 오탐이 많으면 `DEFAULT_CONF`를 높임

권장 방법:

1. 같은 장면에서 `0.35`, `0.50`, `0.60` 비교
2. `/pose` 응답의 `quality_score` 확인
3. 낙상/비낙상 샘플에서 `abnormal_enter` 빈도 비교

## 2. EMA 스무딩

관련 설정:

- `KP_EMA_ENABLE`
- `KP_EMA_ALPHA`
- `KP_MIN_CONF`
- `KP_ALPHA_BY_ID`
- `KP_CONF_HOLD`
- `KP_CONF_MID`
- `KP_ALPHA_MUL_LOW`
- `KP_ALPHA_MUL_MID`
- `KP_ALPHA_MUL_HIGH`
- `KP_JUMP_RATIO`
- `KP_JUMP_RATIO_EXT`

현재 기본 경향:

- 골반/어깨는 더 안정적으로
- 손목/발목은 더 민감하게
- 낮은 confidence에서는 이전 좌표를 더 강하게 유지

튜닝 방향:

- 좌표 떨림이 심하면 알파를 키움
- 반응이 너무 느리면 알파를 낮춤
- 손발이 과하게 튀면 `KP_JUMP_RATIO_EXT`와 `KP_JUMP_ALPHA_MUL` 조정

증상별 힌트:

- 낙상 순간 반응이 늦다
  - `KP_EMA_ALPHA` 약간 낮춤
- 손목/발목이 과하게 튄다
  - `KP_JUMP_RATIO_EXT` 낮추거나 `KP_JUMP_ALPHA_MUL` 높임
- 낮은 confidence 프레임에서 흔들림이 심하다
  - `KP_MIN_CONF` 또는 low confidence multiplier 조정

## 3. Level1 낙상 판단

관련 클래스:

- `Level1Params` in `app/core/level1.py`

중요 파라미터:

- `vy_th`
- `aspect_abs_th`
- `aspect_ratio_th`
- `enter_confirm_s`
- `enter_confirm_need`
- `posture_persist_s`
- `shoulder_drop_persist_s`
- `height_ratio_th`
- `height_persist_s`
- `force_abnormal_aspect_th`
- `force_abnormal_frames`
- `sustain_s`
- `abnormal_timeout_s`
- `recover_s`

### 민감도를 높이고 싶을 때

- `vy_th` 낮춤
- `enter_confirm_s` 낮춤
- `enter_confirm_need` 낮춤
- `sustain_s` 낮춤

부작용:

- 앉기, 줍기, 몸 숙이기 같은 동작에 더 민감해짐

### 오탐을 줄이고 싶을 때

- `enter_confirm_s` 늘림
- `enter_confirm_need` 늘림
- `posture_persist_s` 늘림
- `sustain_s` 늘림

부작용:

- 실제 낙상 반응이 늦어질 수 있음

### 천천히 무너지는 상황을 더 잘 잡고 싶을 때

- `force_abnormal_frames` 줄임
- `force_abnormal_aspect_th`를 약간 완화

부작용:

- 바닥에 오래 앉아 있는 상황도 더 자주 의심할 수 있음

## 4. 재실 감지

관련 클래스:

- `PresenceParams`

현재 기본값:

- `enter_hits=5`
- `exit_hits=10`
- `min_quality=0.10`
- `cool_down_s=0.5`

튜닝 방향:

- enter/exit가 너무 자주 바뀌면 `enter_hits`, `exit_hits`를 늘림
- 사람이 잘 보이는데 enter가 늦으면 `enter_hits`를 줄임

## 5. AI 2차 분류기

관련 위치:

- `app/ai_classifier/inference.py`

현재 기본값:

- `threshold=0.6`
- 최근 30프레임 시퀀스 사용

튜닝 방향:

- AI가 너무 자주 판단을 보류하면 threshold를 낮춤
- AI가 너무 공격적으로 룰베이스를 반려하면 threshold를 높임

주의:

AI 분류기 튜닝은 모델 가중치 품질과 데이터셋 구성의 영향을 크게 받습니다.  
즉, threshold 조정보다 데이터셋 정제가 더 큰 효과를 내는 경우가 많습니다.

## 6. 튜닝 절차 추천

권장 순서:

1. YOLO Pose 안정화
2. EMA 스무딩 조정
3. Level1 규칙 기반 튜닝
4. AI threshold 미세 조정

이 순서를 권장하는 이유는, 상위 입력 품질이 나쁘면 뒤쪽 파라미터를 아무리 만져도 효과가 제한적이기 때문입니다.

## 7. 실험 시 저장하면 좋은 것

- `runs/obs_*.jsonl`
- `runs/clips/fall/*.mp4`
- `runs/clips/not_fall/*.mp4`
- `/health` 상태 스냅샷
- 파라미터 변경 내역

같은 장면을 리플레이로 반복 비교하면 파라미터 영향 파악이 쉬워집니다.

## 8. 추천 비교 시나리오

- 정상 보행
- 천천히 앉기
- 물건 줍기
- 침대/소파로 눕기
- 빠른 낙상
- 천천히 주저앉기
- 근접 상반신 장면
- 부분 가림/측면 진입 장면

이 시나리오들을 같이 봐야 특정 파라미터가 실제 낙상만 민감하게 올리는지, 전체 오탐을 같이 올리는지 판단할 수 있습니다.
