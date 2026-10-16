"""training.experiments - 단계별 개선 효과 측정 (progression study, 오프라인 전용)

규칙만 -> 1D CNN(URFD 원래 방식) -> 데이터 추가 -> 라벨/학습-운영 정렬 -> 모델/특징 변경 -> 임계값 조정을
같은 held-out 데이터, 같은 이벤트 단위 프로토콜(운영과 같은 15Hz obs 재생)로 비교합니다.

- folds.py      : event_group 단위 5-fold 배정 (StratifiedGroupKFold), 저장/불러오기, 학습 fold 안의 검증 분리
- replay.py     : 클립(.npz) -> 운영과 같은 obs 스트림(15Hz, EMA, 정지 꼬리) -> Level1Engine 재생 -> 클립 결과/요약
- verifiers.py  : Level1Engine에 주입하는 검증기 (v1 30프레임 시퀀스, ABNORMAL 진입 사건 구간 v2)

패키지 import만으로는 numpy/torch를 불러오지 않습니다 (표준 라이브러리 테스트 환경 호환).
"""
