"""
검수된 현장 클립으로 배포 v2 분류기(E1, 사건 구간)를 다시 학습해 후보 가중치를 만듭니다.

  python -m training.review_queue list                  # 먼저 검수 (training/review_queue.py)
  python -m training.retrain_from_runs --dry-run        # 할 일만 출력
  python -m training.retrain_from_runs --device 0

단계 (검수 큐에서 fall/not_fall로 검수한 클립만 사용. 미검수, unsure 클립은 쓰지 않음):
1. 검수 큐 export -> 현장 manifest (기본 dataset/_manifest/field_runs.jsonl, fall은 낙상 구간이 있어야 포함)
2. training.datasets.extract_keypoints --rotate-fallback: 기본 학습 키포인트(v3_fb)와 같은 모델/설정으로 추출
   (기본 dataset/_keypoints/field_v3_fb, 이미 추출한 클립은 건너뜀)
   추출 뒤 내보낸 행마다 키포인트 파일이 있는지 확인하고, 없는 클립(영상이 깨짐, 낙상 구간이 클립 끝을 넘음 등)이
   있으면 목록을 출력하고 멈춥니다. 그 클립 없이 학습하려면 --allow-missing.
3. training.train_v2 --extra-manifest: 기본 manifest(공개 데이터) + 현장 행으로 학습, 후보 위치에 저장
   (기본 dataset/_models/candidates/fall_cnn_gap_v2_field_<시각>.pt)
라벨 기록에 읽지 못한 줄이 있으면 실행하지 않습니다 (그 줄이 어떤 클립의 마지막 정정이었는지 알 수 없음).

배포 가중치(app/ai_classifier/weights/, dataset/_models/fall_cnn_gap_v2.pt)는 바꾸지 않습니다. 후보는 평가로
기존 모델보다 나은지 확인한 뒤 사람이 옮기거나 AI_MODEL_PATH로 지정합니다.

이전 동작(runs/clips/not_fall 전체를 검수 없이 라벨 0으로 넣어 v1 30프레임 모델을 재학습)은 없앴습니다.
not_fall에는 AI가 억제한 사건이 들어 있고 그중에는 나중에 LEVEL1로 승격된 실제 낙상도 있으며,
v1 모델은 지금 런타임(v2 사건 구간 분류기)이 쓰지 않습니다. --input-dir/--prefix/--save-dir를 주면 이유를 출력하고 종료합니다.
"""
import argparse
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    # python training/<script>.py 로 직접 실행한 경우에도 저장소 루트의 app/, training/을 찾도록
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import review_queue as RQ
from training.datasets.manifest import write_manifest
from training.paths import REPO_ROOT, RUNS_DIR, is_runtime_weights_path
from training.train_v2 import DEFAULT_KP_DIR, DEFAULT_MANIFEST

DEFAULT_FIELD_KP_DIR = REPO_ROOT / "dataset" / "_keypoints" / "field_v3_fb"
CANDIDATE_DIR = REPO_ROOT / "dataset" / "_models" / "candidates"

REMOVED_V1_MSG = (
    "--input-dir/--prefix/--save-dir (v1 path) were removed: runs/clips/not_fall holds AI-suppressed events, "
    "some of them real falls (later escalated to LEVEL1), so they cannot be labeled 0 without review, and the "
    "runtime no longer loads the v1 model. Review clips with 'python -m training.review_queue' and run "
    "'python -m training.retrain_from_runs' (v2 path) instead."
)


def run_step(name: str, cmd) -> bool:
    """단계 실행. 실패하면 단계 이름과 종료 코드를 출력하고 False."""
    print(f"\n[RUN] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as err:
        print(f"[{name}] failed with exit code {err.returncode}", file=sys.stderr)
        return False
    return True


def kp_path(kp_dir: Path, clip_id: str) -> Path:
    """
    clip_id -> <kp_dir>/<clip_id>.npz. training.datasets.extract_keypoints.clip_out_path와 같은 규칙
    (그 모듈은 numpy를 불러오므로 표준 라이브러리만으로 확인하려고 옮겨 적음).
    """
    parts = [p for p in clip_id.replace("\\", "/").split("/") if p]
    parts = ["".join("_" if c in '<>:"|?*' else c for c in p) for p in parts]
    return kp_dir.joinpath(*parts[:-1], parts[-1] + ".npz")


def missing_keypoints(rows, kp_dir: Path):
    """키포인트 파일이 없는 행 목록."""
    return [r for r in rows if not kp_path(kp_dir, r["clip_id"]).is_file()]


def _abs(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def build_commands(field_manifest: Path, field_kp_dir: Path, manifest: Path, kp_dir: Path, out: Path,
                   model_path=None, device=None, epochs=None):
    """추출, 학습 명령 (저장소 루트에서 실행)."""
    extract = [sys.executable, "-m", "training.datasets.extract_keypoints", "--manifest", str(field_manifest),
               "--out", str(field_kp_dir), "--rotate-fallback"]
    if model_path:
        extract += ["--model", str(model_path)]
    if device is not None:
        extract += ["--device", str(device)]
    train = [sys.executable, "-m", "training.train_v2", "--manifest", str(manifest), "--kp-dir", str(kp_dir),
             "--extra-manifest", str(field_manifest), "--extra-kp-dir", str(field_kp_dir), "--out", str(out)]
    if device is not None:
        train += ["--device", str(device)]
    if epochs is not None:
        train += ["--epochs", str(int(epochs))]
    return extract, train


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Retrain the v2 event-window classifier with reviewed field clips (review queue labels only)"
    )
    parser.add_argument("--runs-dir", default=str(RUNS_DIR), help="Runtime output folder (events/, clips/)")
    parser.add_argument("--labels", default=None, help="Review label log (default: <runs-dir>/review/labels.jsonl)")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Base manifest (public datasets)")
    parser.add_argument("--kp-dir", default=str(DEFAULT_KP_DIR), help="Keypoints of the base manifest")
    parser.add_argument("--field-manifest", default=str(RQ.DEFAULT_EXPORT), help="Exported field manifest (rewritten)")
    parser.add_argument("--field-kp-dir", default=str(DEFAULT_FIELD_KP_DIR), help="Keypoints of field clips")
    parser.add_argument("--out", default=None,
                        help="Candidate weights (default: dataset/_models/candidates/fall_cnn_gap_v2_field_<time>.pt). "
                             "Deployed weight locations are refused")
    parser.add_argument("--model-path", default=None, help="YOLO pose model for extraction (default: extractor default)")
    parser.add_argument("--device", default=None, help="cpu or GPU index for extraction and training")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch cap override (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and commands only")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Train even if some reviewed clips have no keypoints after extraction (they are left out)")
    # 이전 v1 경로의 옵션: 주면 이유를 출력하고 종료
    parser.add_argument("--input-dir", default=None, help="(removed) v1 not_fall folder, see module doc")
    parser.add_argument("--prefix", default=None, help="(removed) v1 array prefix")
    parser.add_argument("--save-dir", default=None, help="(removed) v1 array folder")
    args = parser.parse_args(argv)

    if args.input_dir is not None or args.prefix is not None or args.save_dir is not None:
        print(REMOVED_V1_MSG, file=sys.stderr)
        return 2

    runs_dir = _abs(args.runs_dir)
    try:
        entries, problems = RQ.load_queue(runs_dir, _abs(args.labels) if args.labels else None, strict=True)
    except RQ.LabelLogError as err:
        print(f"Unreadable lines in the review label log: {err}\nFix or delete those lines (a cut-off line may have "
              "been the latest label of a clip) and run again.", file=sys.stderr)
        return 1
    for p in problems:
        print(f"[warn] {p}", file=sys.stderr)
    rows, skipped = RQ.export_rows(entries)
    status = Counter(e["status"] for e in entries)
    n_fall = sum(r["label"] == "fall" for r in rows)
    print(f"[queue] entries {len(entries)}: " + ", ".join(f"{s} {status.get(s, 0)}" for s in RQ.STATUSES))
    print(f"[queue] reviewed rows {len(rows)} (fall {n_fall}, adl {len(rows) - n_fall}); unreviewed and unsure are not used")
    for s in skipped:
        print(f"[skip] {s['id']}: {s['reason']}")
    if not rows:
        print("No reviewed fall/not_fall clips to train on. Label clips first: python -m training.review_queue list",
              file=sys.stderr)
        return 1

    out = _abs(args.out) if args.out else CANDIDATE_DIR / f"fall_cnn_gap_v2_field_{time.strftime('%Y%m%d_%H%M%S')}.pt"
    if is_runtime_weights_path(out):
        print(f"Refusing to write to a deployed weights location: {out}", file=sys.stderr)
        return 2
    if out.exists():
        print(f"Refusing to overwrite existing weights: {out}", file=sys.stderr)
        return 2

    field_manifest, field_kp_dir = _abs(args.field_manifest), _abs(args.field_kp_dir)
    manifest, kp_dir = _abs(args.manifest), _abs(args.kp_dir)
    extract, train = build_commands(field_manifest, field_kp_dir, manifest, kp_dir, out,
                                    model_path=args.model_path, device=args.device, epochs=args.epochs)
    if args.dry_run:
        print("[dry-run] would write", field_manifest)
        for cmd in (extract, train):
            print("[dry-run]", " ".join(cmd))
        return 0
    if not manifest.is_file():
        print(f"Base manifest not found: {manifest}", file=sys.stderr)
        return 1

    write_manifest(field_manifest, rows)
    print(f"[queue] wrote {field_manifest}")
    extract_ok = run_step("extract", extract)
    missing = missing_keypoints(rows, field_kp_dir)
    for r in missing:
        print(f"[missing] {r['clip_id']}: no keypoints ({r['video']})", file=sys.stderr)
    if missing:
        print(f"{len(missing)}/{len(rows)} reviewed clips have no keypoints; reasons are in "
              f"{field_kp_dir / '_extract_log.jsonl'}", file=sys.stderr)
        if not args.allow_missing or len(missing) == len(rows):
            print("Stopping before training. Fix the clips or labels, or pass --allow-missing to train without them.",
                  file=sys.stderr)
            return 1
    elif not extract_ok:
        return 1
    if not run_step("train", train):
        return 1
    print(f"\nCandidate saved: {out}\nThe runtime still loads the deployed weights. Evaluate the candidate first; "
          f"to try it on a device set AI_MODEL_PATH={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
