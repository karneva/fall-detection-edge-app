"""
학습 스크립트의 기본 경로 (저장소 루트 기준, 실행 위치와 무관)

기본값은 기존 스크립트가 쓰던 위치와 같습니다. 학습 데이터 위치는 TRAIN_DATA_DIR로 바꿀 수 있습니다.
가중치는 런타임이 찾는 위치(app/ai_classifier/weights)에 저장해야 장치에서 바로 로드됩니다.
"""
import os
from pathlib import Path

from app.ai_classifier.weights_resolver import DEFAULT_WEIGHTS_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]

# 원본 데이터셋 (URFD 프레임 폴더)
URFD_ROOT = REPO_ROOT / "dataset" / "URFD"

# 전처리된 학습 배열 (X_*.npy, y_*.npy)
DATA_DIR = Path(os.getenv("TRAIN_DATA_DIR", str(REPO_ROOT / "app" / "ai_classifier" / "data")))
ACTIVE_DATA_DIR = DATA_DIR / "active_learning"

# 학습 결과 가중치 (런타임 기본 탐색 위치와 동일)
WEIGHTS_DIR = DEFAULT_WEIGHTS_DIR

# 키포인트 추출에 쓰는 YOLO Pose 모델 (운영과 같은 모델을 써야 함)
YOLO_POSE_MODEL = "yolov8s-pose.pt"

# 운영 중 저장된 오탐 의심 클립
RUNS_NOT_FALL_CLIPS = REPO_ROOT / "runs" / "clips" / "not_fall"
