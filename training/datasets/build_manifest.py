"""
로컬에 받아 둔 공개 데이터셋으로 통합 manifest(JSONL)를 만듭니다 (표준 라이브러리만 사용)

  python -m training.datasets.build_manifest --out dataset/_manifest/manifest_v1.jsonl
  python -m training.datasets.build_manifest --datasets le2i mcfd --out dataset/_manifest/manifest_v1.jsonl

라벨 출처 (행 형식은 training/datasets/manifest.py):
- le2i, caucafall, gmdcsa24, mcfd: OmniFall 구간 라벨 dataset/omnifall_labels/raw/labels__<name>.csv
  (path,label,start,end,subject,cam,dataset; label은 labels__label2id.csv의 id)
  OmniFall 경로를 dataset/<name>/extracted 아래 실제 영상 파일에 맞춥니다. 추측하지 않고,
  못 찾거나 여러 개에 맞는 경로는 요약 파일에 개수와 예시로 남깁니다.
  MCFD는 같은 시나리오를 카메라 8대로 찍었으므로 event_group = mcfd/chuteNN
- urfd: dataset/urfd/raw/urfall-cam0-{falls,adls}.csv 프레임 상태 라벨 (training/datasets/urfd_labels.py),
  영상은 dataset/urfd/raw/<seq>-cam0.mp4
- figshare_ursul29: 폴더(Fall/ADL) 단위 라벨만 있음, segments [] / label_source "folder"

내용이 똑같은 영상(Le2i 번호만 다른 사본, Figshare "(1)" 복사본)은 행은 두고 event_group을 합칩니다.
결과 요약(데이터셋별 행 수, fall/adl 수, 못 맞춘 경로, 중복 영상, 검증 오류)은 manifest 옆
<이름>.summary.json에 기록합니다. dataset/은 git 제외 폴더입니다.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.datasets.fetch_public import SOURCES
from training.datasets.manifest import OMNIFALL_CLASSES, make_row, validate_rows, write_manifest
from training.datasets.urfd_labels import load_urfd_segments
from training.paths import REPO_ROOT

DATASET_ROOT = REPO_ROOT / "dataset"
DEFAULT_OUT = DATASET_ROOT / "_manifest" / "manifest_v1.jsonl"

DATASETS = ("le2i", "caucafall", "gmdcsa24", "mcfd", "urfd", "figshare_ursul29")
# manifest 데이터셋 이름 -> OmniFall 라벨 파일 이름 (labels__<이름>.csv)
OMNIFALL_FILES = {"le2i": "le2i", "caucafall": "caucafall", "gmdcsa24": "GMDCSA24", "mcfd": "mcfd"}
OMNIFALL_LABEL_LICENSE = "CC BY-NC-SA 4.0"
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg"}
URFD_CSVS = ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv")
N_EXAMPLES = 10


def norm(s: str) -> str:
    """경로 조각 비교용 정규화: 소문자, 영숫자 외 문자열은 '_' 하나로 ("video (14)" -> "video_14")."""
    return re.sub(r"[^0-9a-z]+", "_", s.lower()).strip("_")


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def rel_video(path: Path, repo_root: Path) -> str:
    """저장소 루트 기준 상대 경로 (루트 밖이면 절대 경로), 구분자는 '/'."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def list_videos(root: Path) -> list:
    if not root.is_dir():
        return []
    return sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS
                   and not p.name.startswith("._")), key=lambda p: natural_key(p.as_posix()))


def index_videos(root: Path) -> dict:
    """
    영상 파일을 정규화한 키로 색인합니다: {키: set(경로)}.
    키 = 파일 이름(확장자 제외), 상위 폴더 하나 + 파일 이름, 마지막 경로 조각 2~3개.
    (le2i "Coffee_room_01/.../Videos/video (14).avi"는 "coffee_room_01/video_14"로 찾을 수 있음)
    """
    index = {}
    for p in list_videos(root):
        parts = [norm(x) for x in p.relative_to(root).parent.parts]
        stem = norm(p.stem)
        keys = {stem}
        keys.update(f"{a}/{stem}" for a in parts)
        for k in (2, 3):
            if len(parts) >= k - 1:
                keys.add("/".join(parts[len(parts) - (k - 1):] + [stem]))
        for k in keys:
            index.setdefault(k, set()).add(p)
    return index


def omnifall_key(dataset: str, path: str) -> str:
    """OmniFall 경로 -> index_videos 키. CAUCAFall은 파일 이름(HopS1 등)만으로 유일합니다."""
    if dataset == "caucafall":
        return norm(path.rsplit("/", 1)[-1])
    return "/".join(norm(x) for x in path.split("/"))


def read_label2id(path: Path) -> dict:
    """labels__label2id.csv -> {id: 이름}. 없으면 manifest의 OMNIFALL_CLASSES 순서를 씁니다."""
    if not path.exists():
        return dict(enumerate(OMNIFALL_CLASSES))
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["id"]): r["label"].strip() for r in csv.DictReader(f)}


def read_omnifall(csv_path: Path, id2label: dict) -> tuple:
    """
    OmniFall 라벨 CSV -> ({path: {"segments", "subject", "cam"}}, 문제 목록).
    같은 path의 행을 모아 구간 목록으로 만듭니다.
    """
    clips, problems = {}, []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for n, r in enumerate(csv.DictReader(f), 2):
            path = (r.get("path") or "").strip()
            try:
                cls = id2label[int(r["label"])]
                start, end = float(r["start"]), float(r["end"])
            except (KeyError, TypeError, ValueError):
                problems.append(f"line {n}: bad label row {dict(r)}")
                continue
            if not path or not start < end:
                problems.append(f"line {n}: empty path or start >= end ({path!r} {start} {end})")
                continue
            c = clips.setdefault(path, {"segments": [], "subject": r.get("subject"), "cam": r.get("cam")})
            if (r.get("subject"), r.get("cam")) != (c["subject"], c["cam"]):
                problems.append(f"line {n}: subject/cam differ within {path!r}")
            c["segments"].append({"cls": cls, "start_s": start, "end_s": end})
    return clips, problems


def _license(name: str) -> str:
    return SOURCES[name]["license"] if name in SOURCES else ""


def _examples(items) -> list:
    return sorted(items, key=natural_key)[:N_EXAMPLES]


def build_omnifall(name: str, dataset_root: Path, repo_root: Path) -> tuple:
    labels_dir = dataset_root / "omnifall_labels" / "raw"
    csv_path = labels_dir / f"labels__{OMNIFALL_FILES[name]}.csv"
    video_root = dataset_root / name / "extracted"
    stat = {"label_source": "omnifall", "labels": str(csv_path.relative_to(dataset_root).as_posix())}
    if not csv_path.exists() or not video_root.is_dir():
        stat["missing"] = [str(p.relative_to(dataset_root).as_posix())
                           for p in (csv_path, video_root) if not p.exists()]
        return [], stat
    clips, problems = read_omnifall(csv_path, read_label2id(labels_dir / "labels__label2id.csv"))
    index = index_videos(video_root)
    all_videos = set(list_videos(video_root))
    used, rows, unmatched, ambiguous = set(), [], [], []
    for path, c in clips.items():
        hits = index.get(omnifall_key(name, path), set())
        if len(hits) != 1:
            (unmatched if not hits else ambiguous).append(path)
            continue
        video = next(iter(hits))
        used.add(video)
        group = f"{name}/{path.split('/')[0]}" if name == "mcfd" else None
        rows.append(make_row(name, path, rel_video(video, repo_root), c["segments"], "omnifall",
                             f"{_license(name)}; labels: OmniFall {OMNIFALL_LABEL_LICENSE}",
                             subject=c["subject"], cam=c["cam"], event_group=group))
    unlabeled = [v.relative_to(video_root).as_posix() for v in all_videos - used]
    stat.update({
        "label_paths": len(clips),
        "unmatched": len(unmatched), "unmatched_examples": _examples(unmatched),
        "ambiguous": len(ambiguous), "ambiguous_examples": _examples(ambiguous),
        "videos_found": len(all_videos),
        "unlabeled_videos": len(unlabeled), "unlabeled_examples": _examples(unlabeled),
        "label_problems": len(problems), "label_problem_examples": problems[:N_EXAMPLES],
    })
    return rows, stat


# URFD cam0 mp4는 640x240 합성 영상: 왼쪽 절반은 깊이 영상, 오른쪽 절반이 RGB.
# 깊이 영상에서 사람이 검출되면 키포인트가 오염되므로 RGB 절반만 사용
URFD_RGB_CROP = [320, 0, 640, 240]


def build_urfd(dataset_root: Path, repo_root: Path) -> tuple:
    raw = dataset_root / "urfd" / "raw"
    csvs = [raw / f for f in URFD_CSVS]
    stat = {"label_source": "urfd_csv"}
    missing = [p for p in csvs if not p.exists()]
    if missing:
        stat["missing"] = [p.relative_to(dataset_root).as_posix() for p in missing]
        return [], stat
    segs_by_seq = load_urfd_segments(csvs)
    rows, unmatched, no_fall = [], [], []
    for seq in sorted(segs_by_seq, key=natural_key):
        video = raw / f"{seq}-cam0.mp4"
        if not video.exists():
            unmatched.append(seq)
            continue
        segs = segs_by_seq[seq]
        if seq.startswith("fall") and not any(s["cls"] == "fall" for s in segs):
            no_fall.append(seq)
        rows.append(make_row("urfd", seq, rel_video(video, repo_root), segs, "urfd_csv", _license("urfd"),
                             cam="0", crop_px=URFD_RGB_CROP))
    labeled = {f"{s}-cam0.mp4" for s in segs_by_seq}
    unlabeled = [p.name for p in list_videos(raw) if p.name not in labeled]
    stat.update({
        "label_paths": len(segs_by_seq),
        "unmatched": len(unmatched), "unmatched_examples": _examples(unmatched),
        "videos_found": len(list_videos(raw)),
        "unlabeled_videos": len(unlabeled), "unlabeled_examples": _examples(unlabeled),
        "fall_sequences_without_fall_segment": no_fall,
    })
    return rows, stat


_SBJ = re.compile(r"^SBJ_(\d+)_LOC(\d+)$", re.I)
_ACT = re.compile(r"^ACT\d+_", re.I)


def build_figshare(dataset_root: Path, repo_root: Path) -> tuple:
    """
    Figshare 29 Subjects: .../{Fall,ADL}/SBJ_xx_LOCy/ACTnn_<D>_<k>/<시각>.mp4 구조만 받습니다.
    그 밖의 영상(timelapse 등)은 skipped로 남깁니다.
    """
    name = "figshare_ursul29"
    root = dataset_root / name / "extracted"
    stat = {"label_source": "folder"}
    if not root.is_dir():
        stat["missing"] = [root.relative_to(dataset_root).as_posix()]
        return [], stat
    rows, skipped = [], []
    videos = list_videos(root)
    for p in videos:
        parts = p.relative_to(root).parts
        hit = len(parts) - 4  # <Fall|ADL>/<SBJ>/<ACT>/<파일>
        if hit < 0 or parts[hit] not in ("Fall", "ADL") or not _SBJ.match(parts[hit + 1]) \
                or not _ACT.match(parts[hit + 2]):
            skipped.append(p.relative_to(root).as_posix())
            continue
        rel_id = "/".join(parts[hit:-1] + (p.stem,))
        subject = str(int(_SBJ.match(parts[hit + 1]).group(1)))
        rows.append(make_row(name, rel_id, rel_video(p, repo_root), [], "folder", _license(name),
                             subject=subject, label="fall" if parts[hit] == "Fall" else "adl"))
    stat.update({"videos_found": len(videos), "skipped": len(skipped), "skipped_examples": _examples(skipped)})
    return rows, stat


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_duplicate_videos(rows, repo_root: Path) -> list:
    """
    내용이 똑같은 영상 파일(크기가 같은 것끼리만 sha256 비교)을 찾아 같은 event_group으로 묶습니다.
    Le2i에는 같은 영상이 다른 번호로 들어 있고(Home_01/video_2 = video_3 등), Figshare에는
    "이름 (1).mp4" 복사본이 있어 그대로 두면 train/test에 같은 영상이 나뉘어 들어갑니다.
    행과 라벨은 그대로 두고(각각 따로 라벨링됨) event_group만 합칩니다 (rows를 직접 수정).
    반환: 중복 묶음별 clip_id 목록.
    """
    by_size = {}
    for r in rows:
        p = Path(repo_root) / r["video"]
        try:
            by_size.setdefault(p.stat().st_size, []).append((r, p))
        except OSError:
            continue
    dup_sets = []
    for cands in by_size.values():
        if len(cands) < 2:
            continue
        by_hash = {}
        for r, p in cands:
            by_hash.setdefault(_sha256(p), []).append(r)
        dup_sets.extend(rs for rs in by_hash.values() if len(rs) > 1)

    # event_group 합치기 (union-find, 대표 이름은 자연 정렬 첫 번째라 결과가 항상 같음)
    parent = {}

    def find(g):
        parent.setdefault(g, g)
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    for rs in dup_sets:
        roots = sorted({find(r["event_group"]) for r in rs}, key=natural_key)
        for g in roots[1:]:
            parent[g] = roots[0]
    for r in rows:
        if r["event_group"] in parent:
            r["event_group"] = find(r["event_group"])
    return sorted((sorted((r["clip_id"] for r in rs), key=natural_key) for rs in dup_sets),
                  key=lambda ids: natural_key(ids[0]))


def build(datasets=DATASETS, dataset_root: Path = DATASET_ROOT, repo_root: Path = REPO_ROOT) -> tuple:
    """manifest 행 목록과 데이터셋별 요약을 만듭니다: (rows, summary)."""
    dataset_root, repo_root = Path(dataset_root), Path(repo_root)
    rows, per = [], {}
    for name in dict.fromkeys(datasets):  # 같은 이름을 두 번 줘도 한 번만
        if name in OMNIFALL_FILES:
            r, stat = build_omnifall(name, dataset_root, repo_root)
        elif name == "urfd":
            r, stat = build_urfd(dataset_root, repo_root)
        elif name == "figshare_ursul29":
            r, stat = build_figshare(dataset_root, repo_root)
        else:
            raise ValueError(f"unknown dataset {name!r} (choose from {', '.join(DATASETS)})")
        r.sort(key=lambda x: natural_key(x["clip_id"]))
        stat.update({"rows": len(r), "fall": sum(x["label"] == "fall" for x in r),
                     "adl": sum(x["label"] == "adl" for x in r),
                     "event_groups": len({x["event_group"] for x in r})})
        per[name] = stat
        rows.extend(r)
    dups = merge_duplicate_videos(rows, repo_root)
    for name, stat in per.items():
        stat["event_groups"] = len({x["event_group"] for x in rows if x["dataset"] == name})
    errors = validate_rows(rows)
    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rows": len(rows), "fall": sum(x["label"] == "fall" for x in rows),
        "adl": sum(x["label"] == "adl" for x in rows),
        "datasets": per,
        "duplicate_videos": len(dups), "duplicate_video_examples": dups[:N_EXAMPLES],
        "validation_errors": len(errors), "validation_error_examples": errors[:N_EXAMPLES],
    }
    return rows, summary


def summary_path(out: Path) -> Path:
    return out.with_name(out.stem + ".summary.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the unified clip manifest from local public datasets")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    args = ap.parse_args(argv)

    rows, summary = build(args.datasets, args.dataset_root, REPO_ROOT)
    out = args.out if args.out.is_absolute() else Path.cwd() / args.out
    write_manifest(out, rows)
    summary["manifest"] = rel_video(out, REPO_ROOT)
    summary_path(out).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    for name, s in summary["datasets"].items():
        extra = f" MISSING {s['missing']}" if s.get("missing") else ""
        print(f"{name:18s} rows={s['rows']:5d} fall={s['fall']:5d} adl={s['adl']:5d} "
              f"groups={s['event_groups']:5d} unmatched={s.get('unmatched', 0)} "
              f"ambiguous={s.get('ambiguous', 0)} unlabeled_videos={s.get('unlabeled_videos', 0)} "
              f"skipped={s.get('skipped', 0)}{extra}")
    print(f"total rows={summary['rows']} fall={summary['fall']} adl={summary['adl']} "
          f"duplicate_videos={summary['duplicate_videos']} validation_errors={summary['validation_errors']}")
    print(f"-> {out}\n-> {summary_path(out)}")
    for e in summary["validation_error_examples"]:
        print(f"  ! {e}")
    return 0 if not summary["validation_errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
