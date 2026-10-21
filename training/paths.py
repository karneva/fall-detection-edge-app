"""
학습 스크립트의 기본 경로 (저장소 루트 기준, 실행 위치와 무관)

기본값은 기존 스크립트가 쓰던 위치와 같습니다. 학습 데이터 위치는 TRAIN_DATA_DIR로 바꿀 수 있습니다.
가중치는 런타임이 찾는 위치(app/ai_classifier/weights)에 저장해야 장치에서 바로 로드됩니다.
"""
import os
from pathlib import Path

from app.ai_classifier.weights_resolver import DEFAULT_WEIGHTS_DIR, V2_EXTRA_DIRS, V2_WEIGHT_NAME

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

# 운영 출력 폴더 (클립, 이벤트 sidecar)와 검수 라벨 기록 (training/review_queue.py)
RUNS_DIR = REPO_ROOT / "runs"
REVIEW_LABELS = RUNS_DIR / "review" / "labels.jsonl"


def is_runtime_weights_path(path) -> bool:
    """
    런타임이 기본으로 찾는 가중치 위치인지 확인합니다 (weights_resolver 탐색 순서 3, 4).
    런타임 weights/ 폴더 안의 파일, 또는 dataset/_models/의 배포 v2 파일명이면 True.
    재학습 결과를 여기에 쓰면 검증 전에 배포 모델이 바뀌므로 후보 위치에 따로 저장해야 합니다.
    """
    p = os.path.normcase(str(Path(path).resolve()))
    if os.path.normcase(str(Path(p).parent)) == os.path.normcase(str(Path(WEIGHTS_DIR).resolve())):
        return True
    v2 = {os.path.normcase(str((Path(d) / V2_WEIGHT_NAME).resolve())) for d in V2_EXTRA_DIRS}
    return p in v2
