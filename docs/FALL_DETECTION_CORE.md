# Fall Detection Core

## 목표

낙상 감지의 목표는 단순 자세 변화와 실제 사고 가능 상황을 구분하는 것입니다.  
이 프로젝트는 한 프레임의 결과만으로 즉시 결론을 내리지 않고, 시간 축 정보와 상태 전이를 이용합니다.

## 전체 판단 흐름

```text
Pose estimation
  -> Observation
  -> EMA smoothing
  -> Rule-based detection
  -> AI verification
  -> final event
```

## 1. 입력 데이터

낙상 엔진은 `Observation` 구조를 입력으로 받습니다.

중요 필드:

- `tracks[0].bbox`
- `tracks[0].keypoints_smooth`
- `tracks[0].quality_score`
- `target_switched`
- `ts`

현재 구현은 단일 주요 대상 중심입니다.

## 2. 전처리

### 대상 선택

`app/engine/live.py`는 YOLO 결과 중 한 명만 선택합니다.

선택 기준 (`app/engine/target_select.py`, 학습용 추출기와 공유):

- 이전 프레임 바운딩 박스와의 IoU
- 매칭 실패 시 가장 높은 confidence 대상 사용

이 과정에서 대상이 바뀌면 `target_switched=True`가 전달되고, 낙상 엔진은 내부 상태를 리셋합니다.

- 대상 전환 정책 `TARGET_SWITCH_POLICY` (기본 `multi`, R2): 후보가 1명뿐이면 IoU가 낮아도 같은 사람으로 봅니다.
  빠르게 넘어지면 연속 프레임의 상자가 크게 움직여 혼자 있어도 리셋되던 문제를 막습니다. `all`이면 기존 동작입니다.
- 회전 폴백 `ROTATE_FALLBACK` (기본 켜짐): 아무도 못 찾았고 5초 안에 대상을 봤으면 영상을 90/270도로 돌려
  직전 대상 근처의 사람을 다시 찾습니다 (누운 사람 미검출 보완). 폴백으로 찾은 프레임은 대상 전환이 아닙니다.

### EMA 스무딩

스무딩 목적:

- 포즈 좌표 떨림 감소
- 순간 노이즈 억제
- 규칙 기반 판단 안정화

특징:

- 관절별 기본 알파 사용
- 신뢰도 구간별 알파 보정
- 낮은 신뢰도에서는 이전 값 유지 가능
- 급격한 점프 이동은 지연 반영

## 3. 상태 머신

`Level1Engine`은 세 가지 상태를 사용합니다.

### NORMAL

평상시 상태입니다.  
이 단계에서는 낙상 의심에 필요한 복합 신호를 누적합니다.

### ABNORMAL

낙상 의심 상태입니다.  
실제 사고인지, 회복 가능한 일시 동작인지, 혹은 오탐인지 판단하는 중간 단계입니다.

### LEVEL1

낙상 확정 상태입니다.  
이벤트 전송, 클립 저장, 서버 업로드가 이 단계 이후 수행됩니다.

## 4. NORMAL -> ABNORMAL 진입 조건

단일 조건이 아니라 여러 신호를 조합합니다.

### 하강 속도 신호

- 바운딩 박스 중심 `cy`의 증가 속도 사용
- `vy_th`를 넘는 빠른 하강 감지

### 자세 변화 신호

- `aspect = bbox_height / bbox_width`
- 절대 기준 또는 최근 baseline 대비 상대 감소 사용

### 어깨 하강 신호

- 양쪽 어깨 키포인트의 평균 y 좌표 추적
- 상반신 위주 장면에서도 사용 가능

### 높이 붕괴 신호

- 최근 높이 baseline 대비 현재 bbox 높이 감소 여부 확인

### 저자세 지속성

- 낮은 자세가 일정 프레임 이상 유지되면 강제로 `ABNORMAL`
- 급격한 낙상만이 아니라 천천히 주저앉는 상황도 포착하려는 의도

### 부분 신체 감지

화면 하단, 측면, 근접 장면처럼 전신이 다 보이지 않는 경우를 별도로 판단합니다.

### 짧은 미검출 유예 (R1)

- `LOST_GRACE_S`(기본 0.5초)보다 짧게 사람이 검출되지 않으면 워밍업 카운터와 진입 신호 누적을 유지합니다.
- ABNORMAL이면 그동안에도 타임아웃 판정을 이어 갑니다. 더 긴 미검출(화면 밖 등)은 기존처럼 초기화합니다.
- `0`이면 기존 동작 (한 프레임만 놓쳐도 초기화)입니다.

### LEVEL1 뒤 래치 (선택, 기본 꺼짐)

- 기본값(`POST_ALARM_LATCH=0`)에서는 LEVEL1 뒤 누운 채 `abnormal_enter`/`abnormal_exit`가 반복 기록됩니다 (알림은 없음).
  래치를 켜면 반복은 없어지지만, 앉았다/가려진 채 서 있다 다시 넘어지는 경우 등에서 두 번째 알람을 놓칠 수 있어 기본으로 끕니다
  (경우 목록은 [TUNING_GUIDE](TUNING_GUIDE.md)의 'LEVEL1 뒤 래치').
- 켜면: LEVEL1 뒤 엔진은 초기화되지만, 사람이 일어설 때까지 다시 ABNORMAL에 들어가지 않습니다 (복합 신호, 저자세 지속 모두).
- 이전에는 누운 자세가 새 baseline이 되어 `abnormal_enter`/`abnormal_exit`가 몇 초마다 반복됐습니다.
- 일어선 자세 = 넘어지기 전 자세 기준 x 0.8과 `aspect_abs_th`보다 큰 가로세로비. 누적(아니면 1.5배 감쇠)이
  `post_alarm_min_upright_s`(0.4초)에 닿으면 '일어섬'으로 진입 차단을 풀고, `recover_s`(2초)에 닿으면 래치를 지웁니다.
- 일어선 뒤 다시 넘어지면 빠르든 느리든 보통처럼 진입하고, 기준값은 넘어지기 전 값입니다. 차단은 다음 알람에서 다시 걸립니다.
- 회복 종료와 대상 전환에서 래치를 풉니다. `Level1Params.post_alarm_latch=False`면 기존 동작입니다
  (상태도는 [TUNING_GUIDE](TUNING_GUIDE.md)의 'LEVEL1 뒤 래치').

## 5. ABNORMAL 상태에서의 판단

ABNORMAL 상태에서는 아래를 계속 누적합니다.

- 최대 하강 속도
- 최소 aspect
- 평균 품질
- 정지 상태 유지 시간
- 부분 신체 비율

### LEVEL1로 가는 세 가지 경로

#### 1. 정지 상태 지속

낙상 의심 후 움직임이 거의 없는 상태가 `sustain_s` 이상 유지되면 `LEVEL1`로 확정합니다.

#### 2. 타임아웃 확정

`abnormal_timeout_s` 이상 경과했는데도 회복하지 않고, 붕괴가 확인된 상태면 `LEVEL1`로 확정합니다.

1, 2번 경로는 모두 확정 직전에 AI 2차 검증을 거칩니다. (8장 참고)

#### 3. AI 억제 후 승격 (escalation)

AI가 억제한 사건이라도 아래 조건을 모두 만족하면 AI와 무관하게 `LEVEL1`을 냅니다.
이벤트 `reason`은 `escalated_after_ai_suppression`입니다.

- 억제 후 `AI_ESCALATE_S`(기본 60초) 이상 회복하지 않음
- 붕괴 확정(`abnormal_confirmed`)
- 지금도 붕괴 자세(aspect 기준)가 `ai_escalate_collapse_s`(3초) 이상 연속 유지
  (억제 후 일어나 화면 밖으로 나갔다가 서서 돌아온 경우 등의 오알람 방지)
- ABNORMAL 진입 전후 2초 안에 급하강(`vy > vy_th`)이 있었음 (`AI_ESCALATE_REQUIRES_DROP=1`일 때)

## 6. ABNORMAL -> NORMAL 회복

회복 조건은 단일 기준이 아닙니다.

- 자세 비율 회복
- 위쪽 방향 이동
- 중심 이동 발생

이 신호들이 일정 시간 누적되면 `abnormal_exit` 이벤트를 내고 `NORMAL`로 복귀합니다.

## 7. 정지 상태 계산

정지 여부는 두 축을 함께 봅니다.

- bbox 중심 이동량
- 키포인트 평균 이동량

즉, 화면 전체에서 거의 안 움직이면서 포즈도 안정돼야 `still`로 판단합니다.

## 8. AI 2차 검증

`LEVEL1` 확정 직전에 키포인트를 분류기로 검증합니다. 가중치 옆 `<weights>.meta.json`의 `feature_version`으로
두 형식 중 하나를 씁니다 (Day 8).

### v2 (배포 기본): 사건 구간 창

- 모델: Fall1DCNN-GAP (E1), 가중치 `fall_cnn_gap_v2.pt` + `.meta.json`
  (`app/ai_classifier/weights/` -> `dataset/_models/` 순으로 찾고, 메타가 있을 때만 사용)
- 입력: 15Hz 4초 창 (92채널 x 60스텝, 정규화 좌표와 속도 등 v2 특징), 체크포인트의 채널별 mu/sd로 표준화
- 판정 (E*max): ABNORMAL 진입 사건 구간의 첫 창(기준 시각 1.5초 전 ~ 2.5초 후)부터 판정 시점까지 0.5초 간격 4초 창들의
  최대 `p_fall`. 넘어지는 동작이 창 안에 들어갑니다.
- 구현: `app/ai_classifier/event_window.py` (오프라인 `training/experiments/verifiers.py`가 같은 코드를 다시 내보냄),
  기본 검증기 `_ClassifierVerifier`가 엔진에 묶여(`bind_engine`) 엔진의 원시 키포인트 버퍼를 읽음
- 운영 경로와 오프라인 E1max 재생은 같은 가중치와 같은 `Level1Params`(`lost_grace_s`, `post_alarm_latch` 포함)에서
  이벤트, `p_fall`이 비트 단위로 같습니다 (Day 8 점검, 188 클립). 오프라인 실험 재생(`replay.study_params`)은 따로 주지 않으면
  `post_alarm_latch=False`, `lost_grace_s=0.0`으로 고정하므로, 첫 LEVEL1 뒤의 이벤트가 운영 기본값과 다를 수 있습니다.
  v2 가중치, 평가 클립 712개(앞단 multi + 회전 폴백, `lost_grace_s` 0.5)에서 래치 켬/끔 차이는 1클립
  (`n_abnormal`, 첫 알람 뒤 진입 횟수)이고, 클립 단위 검출/오탐 결과는 같습니다 (낙상 379/472, ADL 알람 27/240).

### v1 (legacy): 최근 30프레임

메타가 없거나 `feature_version`이 v1인 가중치. 최근 30프레임 키포인트 시퀀스를 1D CNN으로 검증합니다.

- 30프레임, 프레임당 34차원 좌표값 `(17 keypoints x 2)`
- 좌우 골반 평균을 원점으로 삼아 상대좌표화 (골반이 없으면 대체 기준점)

### 출력 해석

분류기는 낙상 확률 `p_fall`(0~1)을 냅니다 (`FallClassifier.predict_proba`).

- `p_fall < AI_SUPPRESS_PFALL_TH`(기본 0.4): ADL 판정 (경계값 자체는 Fall, Day 8부터 엄격 비교)
- 그 외: Fall 판정
- `AI_SUPPRESS_PFALL_TH <= 0`이면 억제하지 않습니다 (veto 끄기, 판정은 기록).

### AI 게이트 안전 정책

AI는 룰 기반 낙상 알람을 "조용히" 지울 수 없어야 합니다. 이를 위해 아래를 보장합니다.

1. fail-open
   가중치 없음, 로드 실패, torch/numpy import 실패, 추론 예외, `None`/NaN/범위 밖 값, 프레임 부족이면
   룰 결과(낙상)를 그대로 따릅니다. 초기화되지 않은 랜덤 모델은 절대 사용하지 않습니다.
2. 억제는 사건당 1회
   ADL로 판정되면 `ai_suppressed` 이벤트를 내고 ABNORMAL 상태를 유지합니다(suppressed-watch).
   상태 리셋이나 쿨다운 소모가 없고, 같은 사건에서 AI를 다시 호출하지 않습니다.
   - 사람이 일어나면 기존 회복 판정으로 `abnormal_exit`
   - 붕괴가 오래 지속되면 5장의 승격 경로로 `LEVEL1`
3. 모드 선택 (`AI_MODE`)
   - `veto`: ADL 판정 시 억제 (기본)
   - `shadow`: 판정만 기록하고 억제하지 않음 (현장 데이터 수집용)
   - `off`: 분류기를 호출하지 않음
   - 알 수 없는 값은 억제하지 않음

### 이벤트에 남는 AI 정보

`level1`, `ai_suppressed` 이벤트의 `values.ai`에 아래가 기록됩니다.

| 키 | 의미 |
| --- | --- |
| `mode` | veto / shadow / off |
| `decision` | fall / adl / unavailable / error / off |
| `p_fall`, `conf` | 낙상 확률, max(p, 1-p) |
| `suppress` | 실제로 억제했는지 |
| `window_start_ts`, `window_end_ts` | v2: 첫 창 시작(사건 구간 시작) ~ 마지막 버퍼 프레임, v1: 분류한 30프레임 창의 범위 |
| `n_windows` | v2만: 점수를 낸 4초 창 수 |
| `window_starts`, `window_ps` | v2만: 점수를 낸 창들의 시작 시각과 창별 `p_fall` (같은 순서, `p_fall` = 최대값) |
| `argmax_window_start_ts` | v2만: 최대 `p_fall`이 나온 창의 시작 시각 |
| `window_offset_from_abnormal_s` | ABNORMAL 진입 시점 대비 창 시작 오프셋 |
| `latency_ms`, `model_id` | 추론 시간, 가중치 sha256 앞 8자리 |

`ai_suppressed`에는 기존 키 `ai_conf`, `logic`도 그대로 남습니다.

sidecar(`runs/events/*.json`)의 사건 스냅샷 `ai_input`은 모델이 실제로 점수를 낸 입력이고 `kind`로 구분합니다
(승격 이벤트에서도 억제 당시 입력).

| `kind` | 내용 |
| --- | --- |
| `v2_event_windows` | `window_starts`, `window_s`(4초), `ps`(창별 `p_fall`) + 첫 창 시작부터 판정 시점까지의 원시 키포인트 구간 (`kp_buffer.window` 형식: `ts`/`kps`(17 x [x, y, conf])/`bbox`, 앞뒤 `pad_s` 포함) |
| `v1_seq30` | 검증기에 넘긴 최근 30프레임 `ts`/`kps`(평탄화 34값). v1 모델 입력. v2가 점수 낸 창 없이 fail-open한 판정도 이 기록 |

### 알려진 한계

- v1 가중치를 쓰면 CNN이 보는 창은 LEVEL1 시점의 최근 30프레임이라, 넘어지는 동작이 아니라 "누워서 정지"
  구간입니다. v2(기본)는 사건 구간 창이라 이 한계가 없습니다 (Day 8).
- v2도 엔진 버퍼가 30프레임 미만이면 분류기를 부르지 않고 규칙 결과를 따릅니다 (fail-open).
- 사람이 계속 미검출(lost)이거나 품질이 낮으면 승격 조건이 평가되지 않습니다.
- 억제 후 감시 중에 대상 스위칭(`target_switched`) 등으로 `reset_all`이 일어나면 승격 대기도 사라집니다.
  기본 정책(`multi`)에서는 후보가 2명 이상일 때만 일어납니다.
- 급하강 없이 천천히 무너진 낙상을 AI가 억제하면 기본 설정에서는 승격되지 않습니다.
  침대에 눕기 같은 ADL과 구분할 수 없기 때문입니다. v2는 낙상 동작을 창에 넣어 이런 억제를 줄입니다.
- 승격된 알람의 클립은 ABNORMAL 진입 시점 기준으로 자릅니다. 클립 링버퍼(24fps 기준 약 80초)보다
  `ABNORMAL_TIMEOUT_S + AI_ESCALATE_S`가 길면 앞부분이 잘릴 수 있습니다.

## 9. 신뢰도 점수

낙상 확정 시 `confidence_score`를 계산합니다.

구성 요소:

- 신호 강도
  - 최대 하강 속도
  - 최소 aspect
- 정지 상태 비율
- 평균 품질 점수

점수는 이벤트 payload에 포함되어 서버와 하위 장치로 전달됩니다.

## 10. 현재 알고리즘의 특성

강점:

- 단순 임계치 한 개가 아닌 복합 신호 기반
- 순간 노이즈에 덜 민감
- AI 2차 필터 존재
- 천천히 무너지는 상황까지 고려

한계:

- 완전한 멀티인원 추적 시스템은 아님
- 카메라 각도와 화각에 따라 임계값 민감도 변화 가능
- 전신이 잘리지 않는 장면에서 더 안정적
- 데이터셋 품질에 따라 2차 분류기 성능 편차 가능
- 개선 계획은 [ROADMAP.md](ROADMAP.md) 참고
