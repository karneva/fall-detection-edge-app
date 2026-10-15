"""오프라인 평가 도구 (표준 라이브러리만 사용)

- event_metrics: 엔진 이벤트와 정답 라벨을 비교해 이벤트 단위 지표를 계산
- replay_eval: 기록된 obs JSONL(또는 합성 시나리오)을 엔진에 재생해 평가
- compare_results: 두 평가 결과를 비교하고 채택 규칙을 검사
- evalset_tools: 평가셋 manifest 검증(누수 검사)과 요약
- synth: 합성 obs 시나리오 생성기 (회귀 테스트/데이터 없는 평가용)
"""
