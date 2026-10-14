# Roadmap: 1D CNN 2차 검증 고도화 (7일)

## 목표

실제 낙상은 놓치지 않고(높은 recall), 침대에 눕기·빨리 앉기·물건 줍기 같은 하드 ADL 오탐은 줄이는 것.

## 현재 정확도를 막고 있는 원인

코드 분석으로 확인한 원인을 영향이 큰 순서로 정리했습니다.

1. CNN이 낙상 동작을 보지 못함
   AI 검증은 ABNORMAL 진입 후 약 10초가 지난 LEVEL1 시점에, 직전 30프레임만 보고 판단합니다.
   이 30프레임은 넘어지는 동작이 아니라 "누워서 정지" 구간이라, 바닥에 누운 것과 침대에 누운 것을 정적 자세만으로 구분해야 합니다.
2. 학습 라벨 노이즈
   `dataset_builder.py`는 URFD 낙상 영상의 모든 슬라이딩 윈도우를 Fall로 라벨링합니다. 서 있는 구간과 누운 구간도 포함됩니다.
   URFD가 제공하는 프레임 단위 라벨(-1/0/1)을 쓰지 않습니다.
3. 학습 입력과 운영 입력의 불일치
   - URFD는 30fps라 30프레임이 1초이지만, Jetson 운영에서는 2~4초입니다.
   - 운영 입력은 EMA 스무딩된 키포인트, 학습 입력은 원본입니다.
   - 미검출 관절 (0,0)이 골반 기준으로 바뀌면서 큰 이상치가 됩니다.
   - 스케일과 종횡비 정규화가 없습니다.
   - 정규화 코드가 3곳에 복제되어 있습니다.
4. AI가 실제 낙상 알람을 지울 수 있던 경로 (Day 1에서 해결)
   - 가중치 경로가 어긋나면 랜덤 모델로 판정했습니다.
   - 억제 후 쿨다운과 리셋 때문에 누운 사람이 다시 알람을 받지 못했습니다.
5. 액티브 러닝 자기강화 루프
   AI가 억제한 클립을 검수 없이 전부 ADL로 학습합니다. AI가 놓친 실제 낙상도 ADL로 강화됩니다.
6. 평가 체계 부재
   소규모 URFD val 정확도로 모델을 고르고, 이벤트 단위 recall이나 시간당 오탐을 재는 도구가 없습니다.

## 채택 규칙

Day 2 이후 모든 변경은 `replay_eval` 결과로 채택 여부를 정합니다.

- 낙상 이벤트 recall이 떨어지지 않을 것
- 하드 ADL의 시간당 오탐(FA/h)이 줄어들 것
- 창 단위 accuracy는 진단용으로만 사용

## 일정

### Day 1: 안전성과 테스트 기반 ✅

CNN이 실제 낙상 알람을 조용히 지우는 경로를 막았습니다. 분류기의 판정 경계는 바꾸지 않았습니다.

- [x] 룰 엔진을 torch/numpy/cv2 없이 import 가능하게 (테스트 가능성)
- [x] 가중치 경로 해석, 로드 실패 시 비활성화(fail-open), `p_fall` API
- [x] verifier 주입, `AI_MODE`(veto/shadow/off), `p_fall` 임계값 게이트, 이벤트에 AI 판정 기록
- [x] 억제 시 쿨다운 미소모, suppressed-watch, CNN과 무관한 승격(escalation)
- [x] 합성 시나리오 회귀 테스트 (`tests/`)
- [x] 다관점 리뷰 반영: 승격 시 현재 붕괴 자세 확인, 분류기 싱글턴 스레드 안전화, docker-compose 환경변수 전달

직접 확인할 것 (Jetson):

- [ ] 기존 가중치로 기동 후 로그에 `enabled: True`와 `weights_hash`가 찍히는지
- [ ] 가중치 폴더 이름을 잠시 바꾼 뒤 `classifier DISABLED` 로그가 나오고 낙상 시 level1이 정상 송출되는지
- [ ] `AI_MODE=shadow`로 낙상을 연출해 `values.ai.p_fall`이 기록되는지

### Day 2: 측정 체계

판정 로직은 바꾸지 않고, 이후 변경을 숫자로 비교할 수 있게 합니다.

- `tools/replay_eval.py`: 녹화된 `obs_*.jsonl`과 라벨을 받아 엔진을 오프라인으로 재생합니다.
  rules-only와 rules+AI를 비교하고, 이벤트 recall, 시간당 오탐, 알람 지연, 억제된 실제 낙상 수를 냅니다.
- 결과 비교 스크립트와 합성 baseline을 추가하고, eval manifest 예시를 버전 관리합니다.
- `evalset_tools`: train과 eval 사이 subject/session 누수를 error로 처리합니다.
- 타임스탬프가 붙은 원시 키포인트 버퍼(lost 프레임 포함)를 추가하고, ABNORMAL 진입 시점 창을 이벤트에 기록합니다.
- 이벤트 sidecar(CNN 입력 창, `p_fall`, 모델 해시)를 저장하고 처리 fps를 노출합니다.
- [owner] 보유한 obs 로그에 라벨을 달고, 배포 카메라로 연출 낙상과 하드 ADL을 녹화합니다.

### Day 3: 학습과 운영 입력 통일

- `app/ai_classifier/features.py` (numpy만 사용)
  - v1: 기존 정규화를 비트 단위로 그대로 재현
  - v2: 타임스탬프 기준 15Hz 리샘플링, 등방 좌표, 몸통 길이 스케일, 전역 골반 궤적, 속도, conf 마스크
- 정규화 3중 복제를 제거하고, 가중치 `meta.json`(feature_version, arch, hz)을 두어 로드 시 검증합니다.
- 영상별 중간 포맷(npz) 추출 단계를 분리하고, URFD 프레임 라벨 파서를 추가합니다.
- [owner] URFD 중간 포맷 추출 (YOLO만 사용, 학습 없음)

### Day 4: 라벨 정정과 모델 학습 레시피

- 낙상 onset 기준으로 창을 잡아 라벨링하고, 하드 네거티브를 태깅합니다.
- event-group k-fold 분할 (cam0/cam1은 같은 그룹)
- `FallTCN` (dilated residual TCN, 약 40k 파라미터, 입력 길이 무관). 기존 `Fall1DCNN`은 유지합니다.
- 증강: 좌우 flip, 스케일, 회전, jitter, 관절/프레임 dropout, time-warp
- 시드 고정, class-weighted CE, AdamW + cosine, PR-AUC 기준 best 선택, 메트릭 JSON 저장
- temperature scaling과 recall 제약 하의 억제 임계값 보정
- [owner] v2 특징 + FallTCN 학습. v1 baseline과 비교해 라벨 효과와 모델 효과를 분리해서 기록합니다.

### Day 5: 런타임 정렬

- v2 모델은 ABNORMAL 진입 시점 창(낙상 동작 포함)을 분류하게 합니다. v1 가중치는 기존 경로를 유지합니다.
- 알려진 결함 두 가지를 수정합니다 (`tests/test_level1_safety.py`의 `KnownIssuesTest`).
  - CNN 창이 낙상 onset을 포함하지 않는 문제
  - LEVEL1 이후 누운 자세에서 abnormal_enter/exit가 반복되는 문제
- 타깃 연속성 판정 개선: 후보가 1명이면 switch 판정 금지, ABNORMAL 중에는 스위치 리셋 금지
- lost 상태에서도 승격 조건 평가, fps 독립 정지 판정 옵션
- [owner] replay_eval로 rules-only, v1, v2를 비교하고, Jetson에서 shadow 모드로 운영합니다.

### Day 6: 현장 피드백 루프 정상화

- 클립 검수 큐: `runs/clips/{fall,not_fall}`을 사람이 검수한 뒤에만 학습에 사용합니다.
- `retrain_from_runs.py`의 무검수 자동 라벨(0) 제거
  (승격된 실제 낙상 클립도 `not_fall`에 저장되므로 필수입니다)
- 운영 도메인 홀드아웃을 val에 포함하고, recall이 떨어지지 않을 때만 모델을 승격하는 게이트를 둡니다.
- [owner] 클립 검수, fine-tune, 승격 게이트 실행

### Day 7: 데이터 확장과 마무리

- Le2i, UP-Fall 어댑터 (라이선스 확인 후)
- (선택, 플래그) `p_fall` 기반 소프트 융합: 확실한 낙상은 알람을 앞당기고, 애매하면 억제 대신 대기 연장
- 분류기 ONNX export와 parity 테스트, classifier를 포함한 end-to-end 벤치마크
- 최종 평가를 in-domain, cross-dataset, 현장 eval로 나눠 문서에 반영
