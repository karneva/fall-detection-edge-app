import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd):
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Extract not_fall clips from runs/ and retrain the active-learning classifier"
    )
    parser.add_argument(
        "--input-dir",
        default="runs/clips/not_fall",
        help="Folder containing false-positive clips",
    )
    parser.add_argument(
        "--prefix",
        default="not_fall",
        help="Prefix for generated active-learning arrays",
    )
    parser.add_argument(
        "--save-dir",
        default="app/ai_classifier/data/active_learning",
        help="Directory for generated NumPy arrays",
    )
    parser.add_argument(
        "--model-path",
        default="yolov8s-pose.pt",
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
            "app.ai_classifier.video_to_data",
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

    run_step([sys.executable, "-m", "app.ai_classifier.train_active"])


if __name__ == "__main__":
    main()
