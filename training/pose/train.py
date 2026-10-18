"""
포즈 모델 파인튜닝 (ultralytics). 운영과 같은 yolov8s-pose에서 시작해 의사 라벨 데이터셋(dataset.py)으로 추가 학습합니다.

증강: 회전(누운 자세 방향 다양화), 밝기·채도(어두운 실내), 좌우 반전(flip_idx), 기본 mosaic.
검증 손실/AP는 의사 라벨 검증 영상(학습과 다른 영상)으로 보고 best.pt를 고릅니다. 평가 데이터셋은 쓰지 않습니다.

  python -m training.pose.train --data dataset/_pose/v1/data.yaml --epochs 20 --name ft_v1
결과: dataset/_pose/runs/<name>/weights/best.pt -> dataset/_models/yolov8s-pose-<name>.pt로 복사
"""
import argparse
import shutil
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = REPO_ROOT / "dataset" / "_pose" / "runs"

# 증강 기본값 (ultralytics 이름)
AUG = {"degrees": 30.0, "hsv_v": 0.5, "hsv_s": 0.5, "fliplr": 0.5, "mosaic": 1.0, "scale": 0.5, "translate": 0.1}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="yolov8s-pose 파인튜닝")
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default=str(REPO_ROOT / "dataset" / "_models" / "yolov8s-pose.pt"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="ft_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr0", type=float, default=0.002, help="파인튜닝이라 기본(0.01)보다 낮게")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)

    from ultralytics import YOLO
    model = YOLO(args.base)
    model.train(data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=args.device,
                project=str(RUNS), name=args.name, exist_ok=True, seed=args.seed, deterministic=True,
                lr0=args.lr0, workers=args.workers, plots=False, **AUG)
    best = RUNS / args.name / "weights" / "best.pt"
    if best.exists():
        dst = REPO_ROOT / "dataset" / "_models" / f"yolov8s-pose-{args.name}.pt"
        shutil.copy2(best, dst)
        print(f"[pose-train] best -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
