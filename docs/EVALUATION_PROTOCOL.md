# Evaluation Protocol

## 목적

모델을 바꾸거나 튜닝하기 전에, 항상 같은 기준으로 비교할 수 있는 고정 평가셋을 정의합니다.

이 문서의 목표는 두 가지입니다.

1. 모델 변경 전후 성능 비교 기준 고정
2. 데이터 누수와 주관적 평가 방지

## 기본 원칙

고정 평가셋은 학습 데이터와 분리되어야 합니다.

- 학습에 사용한 샘플과 동일한 영상/클립을 평가셋에 넣지 않음
- 같은 원본 영상에서 잘라낸 비슷한 구간을 train과 eval에 동시에 넣지 않음
- 한 번 고정한 평가셋 구성은 실험 중간에 자주 바꾸지 않음

즉, 평가셋은 "모델 개발용 데이터"가 아니라 "비교 기준"입니다.

## 추천 평가 단위

현재 구조에서는 `30프레임 키포인트 시퀀스`가 2차 분류기의 기본 입력이므로, 평가 단위도 아래 둘 중 하나로 고정하는 것이 좋습니다.

### 1. 시퀀스 단위 평가

- 입력: `(34, 30)` 또는 `(30, 34)` 형태의 한 시퀀스
- 장점: 분류기 성능 비교에 직접적
- 단점: 실제 현장 이벤트 수준 해석은 약함

### 2. 이벤트 단위 평가 (1차 지표)

- 입력: 녹화된 obs JSONL 하나 = 낙상 사건 하나 또는 일상행동 구간
- 장점: 실제 운영 체감(놓친 낙상, 시간당 오탐, 알람 지연)과 직접 연결
- 도구: `app.eval.replay_eval` (아래 "이벤트 단위 평가 실행" 참고)

모델/파라미터 채택 여부는 이벤트 단위 지표로 정하고, 시퀀스 단위 지표는 분류기 진단용으로 씁니다.
분류기 창 accuracy가 올라도 운영에서 분류기가 보는 창(사건 시점)과 학습 창이 다르면 효과가 없기 때문입니다.

## 이벤트 단위 평가 실행

### 1. 녹화

장치에서 연출 낙상/하드 ADL을 수행하면서 obs를 기록합니다. 한 파일에 사건 하나가 들어가게 끊어서 녹화합니다.

```bash
curl -X POST http://<device>:8000/record/start
# ... 낙상 또는 ADL 수행 ...
curl -X POST http://<device>:8000/record/stop   # runs/obs_YYYYmmdd_HHMMSS.jsonl
```

### 2. 라벨

파일마다 한 줄씩 라벨 JSONL을 작성합니다. 예시는 `eval/labels.example.jsonl`입니다.

- `label`: `fall` 또는 `adl`
- `fall_start_s`, `fall_end_s`: 파일 첫 프레임 기준 넘어지기 시작/끝난 시각(초). 절대 시각이면 `fall_start_ts`
- `scenario`, `source`, `subject_id`, `session_id`: 집계와 누수 점검용

### 3. 실행과 비교

```bash
# 녹화 데이터 (기본 모드: rules_only, rules_ai)
python -m app.eval.replay_eval --labels eval/labels.jsonl --name field_v1

# 파라미터 바꿔 보기
python -m app.eval.replay_eval --labels eval/labels.jsonl --param ai_escalate_s=30 --name field_v1_esc30

# 비교 (채택 규칙 위반 시 종료 코드 1)
python -m app.eval.compare_results runs/eval/field_v1_<sha>.json runs/eval/field_v1_esc30_<sha>.json

# 데이터 없이 엔진 동작 확인 (합성 시나리오)
python -m app.eval.replay_eval --synthetic
```

모드:

| 모드 | 설명 |
| --- | --- |
| `rules_only` | AI 미사용 |
| `rules_ai` | 현재 가중치로 veto (가중치가 없으면 fail-open이라 rules_only와 동일) |
| `shadow` | AI 판정만 기록 (`files[].ai[].p_fall`로 임계값 분석) |
| `const_fall` / `const_adl` | 항상 낙상/ADL이라고 답하는 가짜 AI (AI가 전부 틀릴 때의 최악 recall 확인) |

### 지표 정의

- 탐지: 낙상 파일에서 `[fall_start - 2s, fall_end + 120s]` 안에 `level1`이 있으면 탐지, 없으면 미탐
- `recall`: 탐지된 낙상 파일 / 낙상 파일
- `fa_per_hour`: ADL 파일의 `level1` 수 / ADL 파일 총 길이(시간)
- `latency_median_s`, `latency_p90_s`: `fall_end`부터 첫 `level1`까지
- `suppressed_true_falls`: 실제 낙상인데 AI가 억제한 파일 수
- `missed_after_suppression`: 그중 승격도 안 되어 끝내 놓친 파일 수
- `detected_by_escalation`: 승격으로 잡은 낙상 수

### 채택 규칙

`compare_results`가 아래를 검사하고, 하나라도 어기면 종료 코드 1(REJECT)을 냅니다.

1. 모든 모드에서 `recall`이 baseline보다 떨어지지 않을 것
2. `fa_per_hour`가 늘지 않을 것 (`--allow-fa-increase`로 완화)
3. baseline에서 잡던 낙상 파일을 새로 놓치지 않을 것 (집계 recall이 같아도 파일 단위로 확인)
4. 비교가 성립할 것: 같은 모드 집합, 같은 데이터셋 지문(라벨과 obs 파일 내용 해시), 같은 채점 기준,
   같은 분류기 로드 여부 (데이터셋이 다른 비교는 `--allow-dataset-mismatch`로만 허용)

그다음 `latency`, `suppressed_true_falls` 순으로 비교합니다. 파라미터와 가중치 해시 차이는 함께 출력됩니다.

주의:

- `rules_ai`, `shadow`는 분류기가 로드되지 않으면 실행을 거부합니다(`--allow-disabled-ai`로 강제 실행).
- `--no-files`로 만든 결과는 3번(파일 단위) 검사를 할 수 없습니다.

`eval/baselines/synthetic_v0.json`은 합성 시나리오 baseline이며, 테스트(`tests/test_eval.py`)가
엔진 변경 시 이 baseline 대비 회귀가 없는지 자동으로 확인합니다.

### 현장 데이터 자동 수집 (sidecar)

장치는 `level1`/`ai_suppressed`가 날 때마다 `runs/events/*.json`을 남깁니다.

- ABNORMAL 진입 기준 `[-1.5s, +2.5s]` 원시 키포인트 (낙상 동작 포함, 미검출 프레임은 `null`)
  - 저자세 지속(`enter_reason=low_posture_sustained`)으로 진입했다면 저자세가 시작된 시각(`event_anchor_ts`)부터
- AI가 판정할 때 실제로 본 30프레임(승격 이벤트에서도 억제 당시 입력), `p_fall`, 가중치 해시
- 엔진 파라미터, 실측 FPS, 클립 경로
- `review.human_label`: 검수 전 `pending` (Day 8 검수 큐에서 사용)

AI가 억제한 사건(`not_fall` 클립)도 검수 전에는 ADL로 단정하지 않습니다.

## 평가셋 구성 권장 시나리오

고정 평가셋에는 아래 시나리오를 균형 있게 포함하는 것이 좋습니다.

### Fall

- 빠른 낙상
- 천천히 주저앉기
- 부분 가림이 있는 낙상
- 카메라 근접 상태의 낙상

### ADL

- 걷기
- 서 있기
- 앉기
- 침대/소파에 눕기
- 물건 줍기
- 몸 숙이기
- 고개 흔들기 / 상체만 흔들기
- 화면 가장자리에서의 부분 진입

핵심은 단순 ADL보다 `오탐을 유발하는 hard negative`를 많이 넣는 것입니다.

## 데이터 출처 구성

현재 프로젝트는 아래 두 축을 같이 쓰는 것이 적절합니다.

- `URFD`
- 직접 촬영 데이터

권장 비율은 고정된 정답이 있는 것은 아니지만, 평가셋은 아래 관점을 만족해야 합니다.

- 공개 데이터셋 일반화 성능 확인
- 실제 사용 환경 적합성 확인

따라서 평가셋 summary를 낼 때는 전체 정확도만 보지 말고 최소한 아래 둘을 분리해서 봐야 합니다.

- `source=urfd`
- `source=custom`

## 고정 평가셋 분리 원칙

### 절대 섞으면 안 되는 경우

- 같은 원본 영상에서 나온 train sample과 eval sample
- 같은 사람, 같은 복장, 같은 위치, 같은 촬영 세션의 연속 샘플을 train/eval로 분산

### 권장 분리 기준

- 영상 파일 단위 분리
- 촬영 세션 단위 분리
- 가능하면 사람(subject) 단위도 분리

## Manifest 기반 관리

고정 평가셋은 폴더명만으로 관리하지 말고 manifest 파일로 관리하는 것을 권장합니다.

이 저장소에는 예시 파일과 검증 스크립트를 함께 제공합니다.

- 예시 manifest:
  `eval/eval_manifest.example.jsonl`
- 검증/요약 스크립트:
  `app/eval/evalset_tools.py`

검증 규칙:

- 학습 split(`train`, `val`)과 평가 split(`test`, `eval_fixed`) 사이에 같은 `session_id`나 `subject_id`가 있으면 error
  (`--allow-leak`으로 warning 완화 가능, 단 그 결과는 보고서에 명시)
- URFD처럼 한 사건을 여러 카메라로 찍은 경우(cam0/cam1) 같은 `session_id`를 써야 누수 검사가 동작
- `--check-files`로 `file_path` 존재 여부까지 확인 (상대경로는 `--root` 기준, 기본값은 현재 디렉터리이므로 저장소 루트에서 실행)

```bash
python -m app.eval.evalset_tools validate --manifest eval/eval_manifest.example.jsonl --check-files --root .
python -m app.eval.evalset_tools summary --manifest eval/eval_manifest.example.jsonl
```

## Manifest 필드 규격

각 샘플은 JSONL 한 줄로 기록합니다.

필수 필드:

- `sample_id`
- `split`
- `source`
- `label`
- `scenario`
- `subject_id`
- `session_id`
- `file_path`

권장 필드:

- `camera_view`
- `notes`

## 필드 의미

- `sample_id`
  샘플 고유 ID
- `split`
  `train`, `val`, `test`, `eval_fixed` 중 하나
- `source`
  `urfd`, `custom` 등 데이터 출처
- `label`
  `fall` 또는 `adl`
- `scenario`
  `fast_fall`, `sit_down`, `pick_up`, `lie_down` 등 행동 유형
- `subject_id`
  사람 식별자
- `session_id`
  촬영 세션 또는 원본 영상 그룹 ID
- `file_path`
  원본 파일 또는 파생 샘플 경로

## 고정 평가셋 운영 규칙

1. `eval_fixed`에 들어간 샘플 목록은 버전 관리한다.
2. 새 데이터를 추가하더라도 기존 `eval_fixed`는 쉽게 바꾸지 않는다.
3. 모델 실험 결과는 항상 같은 `eval_fixed` 기준으로 비교한다.
4. 필요하면 `eval_fixed_v1`, `eval_fixed_v2`처럼 버전만 올린다.

## 최소 보고 지표

이벤트 단위(1차): `recall`, `fa_per_hour`, `latency_median_s`, `suppressed_true_falls`를 모드별, scenario별로 남깁니다.

시퀀스 단위(진단용)로는 최소한 아래 지표를 남깁니다.

- 전체 accuracy
- fall precision / recall
- adl precision / recall
- source별 accuracy
- scenario별 accuracy

현재 단계에서는 confusion matrix까지 같이 남기면 더 좋습니다.

## 이번 단계의 목표

이번 고도화 단계에서 가장 먼저 해야 할 일은 다음입니다.

1. 직접 촬영 샘플과 URFD 샘플을 manifest로 정리
2. `eval_fixed`로 사용할 샘플을 고정
3. manifest 검증 스크립트로 중복/필수 필드 누락 점검
4. baseline 모델을 이 평가셋으로 기록
