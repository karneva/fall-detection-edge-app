"""
운영 중 저장된 not_fall 클립을 학습 배열로 변환하고 액티브 러닝 재학습까지 실행합니다.

  python -m training.retrain_from_runs
"""
import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    # python training/<script>.py 로 직접 실행한 경우에도 저장소 루트의 app/, training/을 찾도록
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.paths import ACTIVE_DATA_DIR, RUNS_NOT_FALL_CLIPS, YOLO_POSE_MODEL


def run_step(cmd):
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Extract not_fall clips from runs/ and retrain the active-learning classifier"
    )
    parser.add_argument(
        "--input-dir",
        default=str(RUNS_NOT_FALL_CLIPS),
        help="Folder containing false-positive clips",
    )
    parser.add_argument(
        "--prefix",
        default="not_fall",
        help="Prefix for generated active-learning arrays",
    )
    parser.add_argument(
        "--save-dir",
        default=str(ACTIVE_DATA_DIR),
        help="Directory for generated NumPy arrays",
    )
    parser.add_argument(
        "--model-path",
        default=YOLO_POSE_MODEL,
        help="YOLO pose model path used for extraction",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    run_step(
        [
            sys.executable,
            "-m",
            "training.video_to_data",
            "--input-dir",
            str(input_dir),
            "--label",
            "0",
            "--prefix",
            args.prefix,
            "--save-dir",
            args.save_dir,
            "--model-path",
            args.model_path,
        ]
    )

    run_step([sys.executable, "-m", "training.train_active"])


if __name__ == "__main__":
    main()
