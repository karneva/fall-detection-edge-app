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

### 1-1. 앞단 개선 (R1, R2, 회전 폴백)

Day 8부터 세 가지 모두 기본으로 켜져 있습니다. 끄면 Day 7 이전 동작입니다.

| 변수 | 기본값 | 끄는 값 | 설명 |
| --- | --- | --- | --- |
| `LOST_GRACE_S` | `0.5` | `0` | R1. 이 시간(초)보다 짧은 미검출에는 워밍업과 진입 신호 누적을 유지하고, ABNORMAL이면 타임아웃 판정을 이어감 |
| `TARGET_SWITCH_POLICY` | `multi` | `all` | R2. 사람 후보가 2명 이상일 때만 "다른 사람으로 바뀜"(엔진 리셋)으로 봄. 혼자 빠르게 넘어져 상자가 크게 움직여도 리셋하지 않음 |
| `ROTATE_FALLBACK` | `1` | `0` | 사람을 놓친 프레임에서만 영상을 90/270도로 돌려 직전 대상 근처를 다시 검출 (그 프레임만 추론 2회 추가) |

켠 근거 (Day 7, 운영과 같은 재생 조건 5-fold, E0max 분류기, [MODEL_STUDY](MODEL_STUDY.md#앞단-개선-엔진-처리와-회전-폴백-day-7)):

| 단계 | 탐지율 | ADL 오탐률 | Balanced |
| --- | --- | --- | --- |
| R1만 (운영과 같은 전환) | 0.625 | 0.110 | 0.732 |
| + R2 | 0.717 | 0.118 | 0.789 |
| + 회전 폴백 | 0.806 | 0.150 | 0.827 |

- 채택 규칙: 5개 fold 모두 탐지율이 오르고, ADL 오탐률 증가(pooled)가 0.05 이하면 켬. 놓친 낙상이 오탐보다 비싸다는 판단입니다.
  측정을 본 뒤 정한 규칙(사후)이라 미리 정한 기준은 아닙니다.
- 회전 폴백의 오탐률 증가는 pooled +0.032로 기준 안이지만, fold별로는 최대 +0.064입니다. 바닥에 눕는 일상 동작에서 늘어납니다.
- 끄는 경우: 오탐이 문제가 되는 설치(바닥 매트에서 자주 눕는 곳 등)는 `ROTATE_FALLBACK=0`, 여러 사람이 자주 겹치는 곳에서 대상이
  다른 사람으로 넘어가는데도 리셋이 안 되는 문제가 보이면 `TARGET_SWITCH_POLICY=all`.
- 오프라인 실험(`training/experiments/replay.study_params`)은 이 환경변수를 쓰지 않고 이전 단계 결과가 재현되도록 값을 고정합니다.

## 2. EMA 스무딩

관련 설정:

- `KP_EMA_ENABLE` (환경변수)
- `KP_MIN_CONF` (환경변수)
- 아래는 `app/config.py` 상수
- `KP_ALPHA_BY_ID`, `KP_ALPHA_DEFAULT`
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
  - `KP_ALPHA_BY_ID`(골반/어깨)와 `KP_ALPHA_DEFAULT`를 약간 낮춤
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
- `lost_grace_s` (환경변수 `LOST_GRACE_S`, 위 1-1)
- `post_alarm_latch`, `post_alarm_min_upright_s`

### LEVEL1 뒤 래치 (`post_alarm_latch`, 환경변수 `POST_ALARM_LATCH`, 기본 꺼짐)

기본으로 끈 이유 (Day 8 적대적 검증): 래치는 일어선 자세로만 풀리고 만료가 없어서, 아래 경우 두 번째 낙상의 알람을 놓쳤습니다
(래치를 끄면 모두 알람이 납니다).

- 알람 뒤 의자·침대 가장자리에 앉아 있다가 다시 넘어짐 (앉은 자세는 '일어섬'이 아님)
- 하체가 가려진 채 서 있거나, 카메라 각도 때문에 서 있는 bbox가 넓게 보임 (넘어지기 전 기준의 0.8배를 못 넘음)
- 쓰러진 사람이 나간 뒤 오래 비었다가 다른 사람이 혼자 들어와 넘어짐 (대상 전환이 일어나지 않아 래치가 남음)
- 쿨다운(25초) 중에 일어섰다가 쿨다운이 끝날 무렵 다시 넘어짐 (쿨다운 중에는 일어섬을 보지 않음)
- 0.4초 미만으로 겨우 일어섰다 바로 다시 무너짐

래치를 끄면 LEVEL1 뒤 누운 채 `abnormal_enter`/`abnormal_exit`가 몇 초마다 반복 기록되지만, 이 이벤트는 로그만 남고 알림은 나가지 않습니다.
놓친 낙상의 비용이 더 커서 끈 쪽을 기본값으로 했습니다. 켤 때의 동작은 아래와 같습니다.

- 문제: LEVEL1 뒤 엔진을 초기화하면 누운 자세가 새 기준이 되어, 계속 누워 있는데도 `abnormal_enter` -> `abnormal_exit(recovered)`가
  몇 초마다 반복됐습니다 (알람은 다시 안 나가고 이벤트만 반복).
- 일어선 자세 = 가로세로비 > 넘어지기 전 기준 x 0.8, 그리고 > `aspect_abs_th`. 알람 뒤 NORMAL의 매 관측 프레임에서
  일어선 자세면 누적, 아니면 1.5배로 감쇠합니다 (프레임당 최대 0.25초, 오래 미검출된 뒤 첫 프레임 한 번으로 쌓이지 않게).
- 상태 (LEVEL1이 날 때마다 1부터 다시):
  1. 차단 (알람 직후): 복합 신호 진입과 저자세 지속 진입을 모두 막습니다. 바닥에서 움직여 신호가 나도 같은 사건으로 봅니다.
  2. 일어섬 (누적 >= `post_alarm_min_upright_s`, 기본 0.4초): 진입 차단을 풉니다. 다시 넘어지면 빠르든 느리든
     (저자세 지속 경로 포함) 보통처럼 진입하고, 진입 기준값은 누운 자세가 아니라 넘어지기 전 값을 씁니다.
     다음 알람이나 해제 전까지 이 상태를 유지합니다 (서 있던 시간이나 다시 넘어지는 데 걸린 시간과 무관).
  3. 해제 (누적 >= `recover_s`, 2초): 래치를 지웁니다 (기준값도 지움). 회복 종료(`abnormal_exit`)와 대상 전환에서도 해제됩니다.
- `post_alarm_min_upright_s`: 낮추면 누운 채 bbox가 잠깐 세로로 튀는 잡음에도 차단이 풀려 같은 사람에게 알람이 한 번 더 날 수 있고,
  높이면 아주 잠깐 일어섰다 다시 넘어지는 경우를 놓칠 수 있습니다. 합성 시험(10/15/30fps)에서 0.3초 서 있음 + 0.5~5초 붕괴는
  모두 두 번째 알람, 누운 채 1~3프레임 튐과 바닥에서 몸부림은 재진입 없음.
- 끄면(`post_alarm_latch=False`) 기존 동작입니다. 오프라인 실험 재생(`replay.study_params`)은 재현을 위해 끈 채로 고정합니다.

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
- `app/core/level1.py` (`_verify_with_ai`, 승격 로직)

환경변수:

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `AI_MODE` | `veto` | `veto` 억제 / `shadow` 기록만 / `off` 미사용 |
| `AI_MODEL_PATH` | (빈 값) | 가중치 경로. 비우면 `app/ai_classifier/weights/`의 `fall_model_active_best.pth` → `fall_model_best.pth` 순서로 탐색 |
| `AI_SUPPRESS_PFALL_TH` | `0.4` | `p_fall`이 이 값 미만일 때만 억제 (경계값 0.4 자체는 알람). 0 이하면 억제하지 않음 |
| `AI_ESCALATE_S` | `60.0` | 억제 후 이 시간 이상 붕괴가 지속되면 알람 승격 |
| `AI_ESCALATE_REQUIRES_DROP` | `1` | 승격에 진입 시점 급하강을 요구할지 여부 |

튜닝 방향:

- AI가 실제 낙상을 너무 자주 억제하면 `AI_SUPPRESS_PFALL_TH`를 낮춤 (예: 0.2)
- 하드 ADL 오탐이 여전히 많으면 임계값을 올리기 전에 shadow 모드로 `p_fall` 분포부터 확인
- 억제된 실제 낙상의 알람 지연을 줄이려면 `AI_ESCALATE_S`를 낮춤 (오탐 증가와 교환)

가중치 확인:

- 시작 로그 `[AI] mode=... status={'enabled': True, 'weights_hash': ...}`로 로드 여부를 확인합니다.
- `enabled: False`면 AI 없이 룰 기반으로만 동작합니다 (알람은 정상 송출).
- `/health`, `/level1/status` 응답의 `ai_classifier` 항목에서도 같은 정보를 볼 수 있습니다.
  (분류기가 아직 로드되기 전이면 `state: not_loaded`)
- 가중치 옆의 `<weights>.meta.json`(학습 시 자동 생성)에 입력 특징 버전, 구조, 학습 데이터가 기록됩니다.
  런타임이 처리할 수 없는 조합(예: `feature_version: v2` 가중치)이면 분류기를 끄고(fail-open) `load_error`에 이유를 남깁니다.
  meta가 없는 기존 가중치는 v1로 간주합니다.

shadow 모드로 현장 데이터 모으기:

1. `AI_MODE=shadow`로 실행합니다. 알람은 룰 기반 그대로 나갑니다.
2. `level1` 이벤트의 `values.ai.p_fall`과 실제 상황(낙상/오탐)을 함께 기록합니다.
3. 실제 낙상의 `p_fall`이 임계값 아래로 내려가는 사례가 없을 때 `veto`로 전환합니다.

주의:

AI 분류기 튜닝은 모델 가중치 품질과 데이터셋 구성의 영향을 크게 받습니다.  
즉, threshold 조정보다 데이터셋 정제가 더 큰 효과를 내는 경우가 많습니다.

## 6. 튜닝 절차 추천

권장 순서:

1. YOLO Pose 안정화
2. EMA 스무딩 조정
3. Level1 규칙 기반 튜닝
4. AI 임계값(`AI_SUPPRESS_PFALL_TH`) 미세 조정

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
