"""training - 오프라인 학습/데이터 준비 도구 (GPU 학습 머신용, Jetson 런타임은 import하지 않음)

실행 예 (저장소 루트에서):
  python -m training.dataset_builder
  python -m training.train
  python -m training.review_queue list      # 현장 클립 검수 (표준 라이브러리만 사용)
  python -m training.retrain_from_runs      # 검수된 현장 클립으로 v2 후보 가중치 학습
"""
