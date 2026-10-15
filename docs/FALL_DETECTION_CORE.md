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

선택 기준:

- 이전 프레임 바운딩 박스와의 IoU
- 매칭 실패 시 가장 높은 confidence 대상 사용

이 과정에서 대상이 바뀌면 `target_switched=True`가 전달되고, 낙상 엔진은 내부 상태를 리셋합니다.

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

`LEVEL1` 확정 직전에 최근 30프레임 키포인트 시퀀스를 1D CNN으로 검증합니다.

### 입력

- 30프레임
- 프레임당 34차원 좌표값 `(17 keypoints x 2)`

### 정규화

- 좌우 골반 평균을 원점으로 삼아 상대좌표화
- 골반이 없으면 대체 기준점 사용

### 출력 해석

분류기는 낙상 확률 `p_fall`(0~1)을 냅니다 (`FallClassifier.predict_proba`).

- `p_fall <= AI_SUPPRESS_PFALL_TH`(기본 0.4): ADL 판정
- 그 외: Fall 판정

기본값 0.4는 기존 규칙(`ADL로 판정 & softmax 최댓값 >= 0.6`)과 동치입니다.

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
| `window_start_ts`, `window_end_ts` | 분류한 30프레임 창의 시간 범위 |
| `window_offset_from_abnormal_s` | ABNORMAL 진입 시점 대비 창 시작 오프셋 |
| `latency_ms`, `model_id` | 추론 시간, 가중치 sha256 앞 8자리 |

`ai_suppressed`에는 기존 키 `ai_conf`, `logic`도 그대로 남습니다.

### 알려진 한계

- CNN이 보는 창은 LEVEL1 시점의 최근 30프레임이라, 넘어지는 동작이 아니라 "누워서 정지" 구간입니다.
  (`window_offset_from_abnormal_s`가 보통 `sustain_s - 2`초 이상) 로드맵 Day 7(런타임 정렬)에서 해결합니다.
- LEVEL1 이후 리셋되면 누운 자세가 새 baseline이 되어 `abnormal_enter`/`abnormal_exit`가 몇 초 주기로 반복됩니다.
- 사람이 계속 미검출(lost)이거나 품질이 낮으면 승격 조건이 평가되지 않습니다.
- 억제 후 감시 중에 대상 스위칭(`target_switched`) 등으로 `reset_all`이 일어나면 승격 대기도 사라집니다. (Day 7)
- 급하강 없이 천천히 무너진 낙상을 AI가 억제하면 기본 설정에서는 승격되지 않습니다.
  침대에 눕기 같은 ADL과 구분할 수 없기 때문이며, 낙상 동작을 보는 Day 7의 창 정렬로 해결합니다.
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
