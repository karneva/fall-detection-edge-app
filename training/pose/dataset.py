"""
포즈 파인튜닝 데이터: 평가에 쓰지 않는 영상에서 프레임을 뽑아 교사 모델 + 회전 TTA로 의사 라벨을 달고
YOLO pose 학습 형식으로 저장합니다 (ultralytics, cv2 필요).

출처 (평가 5개 데이터셋과 겹치지 않음 -> progression 평가와 누수 없음)
  - WanFall 합성 (dataset/wanfall_synthetic, 클래스별 폴더): 누운 자세 위주(fallen, lying, fall, lie_down)
    + 서 있는 자세 일부(standing, walking, sitting, sit_down, stand_up, other: 기존 능력 유지용)
  - Figshare 29 Subjects (manifest의 figshare_ursul29, 약 1.9초 클립): 클립 중간과 끝

저장 형식 (out_dir)
  images/{train,val}/<id>.jpg, labels/{train,val}/<id>.txt, data.yaml, items.json (프레임 출처와 라벨 통계)
  라벨 한 줄 = 사람 1명: "0 cx cy w h" + 관절 17개 "x y v" (정규화, v = 2 보임 / 0 안 보임)
  한 프레임의 교사 검출 중 하나라도 품질 기준(keep_person)을 못 넘으면 그 프레임은 버림
  (라벨이 없는 사람이 배경으로 학습되지 않도록). 교사가 아무도 못 찾은 프레임도 버림.
  학습/검증 분할은 영상 단위 (같은 영상의 프레임은 한쪽에만).
"""
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WANFALL_DIR = REPO_ROOT / "dataset" / "wanfall_synthetic" / "extracted" / "omnifall-synthetic_av1"
DEFAULT_OUT = REPO_ROOT / "dataset" / "_pose" / "v1"

# COCO-17 좌우 반전 짝 (ultralytics fliplr 증강용)
COCO_FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# WanFall 클래스별 (영상 수, 영상당 프레임 위치 비율). 누운 자세 위주, 선 자세는 기존 능력 유지용으로 조금
WANFALL_PLAN: Dict[str, Tuple[int, Tuple[float, ...]]] = {
    "fallen": (600, (0.4, 0.9)),
    "lying": (600, (0.3, 0.8)),
    "fall": (400, (0.7, 0.95)),
    "lie_down": (300, (0.6, 0.95)),
    "sit_down": (150, (0.8,)),
    "sitting": (150, (0.5,)),
    "standing": (150, (0.5,)),
    "walking": (150, (0.5,)),
    "stand_up": (150, (0.3,)),
    "other": (150, (0.5,)),
}
FIGSHARE_POS = (0.5, 0.95)

__all__ = ["COCO_FLIP_IDX", "WANFALL_PLAN", "plan_wanfall", "plan_figshare", "split_by_video", "yolo_label_lines",
           "read_frames", "build_dataset", "write_data_yaml"]

Item = Dict[str, Any]   # {"video": 경로, "frac": 영상 길이 비율, "group": 영상 id, "source": ..., "cls": ...}


def plan_wanfall(root: Path = WANFALL_DIR, plan: Dict[str, Tuple[int, Tuple[float, ...]]] = WANFALL_PLAN,
                 seed: int = 0) -> List[Item]:
    """클래스 폴더마다 영상 n개를 seed로 고르고, 영상마다 지정 비율 위치의 프레임 항목을 만듦"""
    rng = random.Random(seed)
    items: List[Item] = []
    for cls, (n, fracs) in plan.items():
        vids = sorted((Path(root) / cls).glob("*.mp4"))
        for v in sorted(rng.sample(vids, min(n, len(vids)))):
            for f in fracs:
                items.append({"video": str(v), "frac": float(f), "group": f"wanfall/{cls}/{v.stem}",
                              "source": "wanfall", "cls": cls})
    return items


def plan_figshare(manifest: Path, fracs: Sequence[float] = FIGSHARE_POS) -> List[Item]:
    """manifest의 Figshare 클립마다 지정 비율 위치의 프레임 항목"""
    items: List[Item] = []
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("dataset") != "figshare_ursul29":
            continue
        for f in fracs:
            items.append({"video": str(REPO_ROOT / r["video"]), "frac": float(f), "group": r["clip_id"],
                          "source": "figshare", "cls": r.get("label")})
    return items


def split_by_video(items: Sequence[Item], val_frac: float = 0.1, seed: int = 0) -> Dict[str, str]:
    """영상(group) 단위 학습/검증 분할 -> {group: "train" | "val"}"""
    groups = sorted({it["group"] for it in items})
    rng = random.Random(seed)
    val = set(rng.sample(groups, int(round(len(groups) * float(val_frac)))))
    return {g: ("val" if g in val else "train") for g in groups}


def yolo_label_lines(persons: Sequence[Dict[str, Any]], w: int, h: int, vis_th: float = 0.5) -> List[str]:
    """사람 목록 (원본 픽셀 좌표) -> YOLO pose 라벨 줄. 상자는 영상 안으로 자름, 신뢰도 < vis_th 관절은 0 0 0"""
    lines = []
    for p in persons:
        x1, y1, x2, y2 = (float(v) for v in p["box"])
        x1, x2 = max(0.0, min(x1, w - 1.0)), max(0.0, min(x2, w - 1.0))
        y1, y2 = max(0.0, min(y1, h - 1.0)), max(0.0, min(y2, h - 1.0))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        vals = [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]
        for x, y, c in np.asarray(p["kps"], dtype=np.float64):
            inside = 0 <= x < w and 0 <= y < h
            if c >= vis_th and inside:
                vals += [x / w, y / h, 2]
            else:
                vals += [0.0, 0.0, 0]
        lines.append("0 " + " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in vals))
    return lines


def read_frames(video: str, fracs: Sequence[float]) -> Dict[float, np.ndarray]:
    """영상에서 길이 비율 위치의 프레임 (순차 읽기: 코덱마다 seek가 부정확할 수 있음)"""
    import cv2
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        cap.release()
        return {}
    want = {min(n - 1, max(0, int(round(f * (n - 1))))): f for f in fracs}
    out, i, last = {}, 0, max(want)
    while i <= last:
        ok, img = cap.read()
        if not ok:
            break
        if i in want:
            out[want[i]] = img
        i += 1
    cap.release()
    return out


def write_data_yaml(out_dir: Path) -> Path:
    p = Path(out_dir) / "data.yaml"
    p.write_text(
        f"path: {Path(out_dir).resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        f"kpt_shape: [17, 3]\nflip_idx: {COCO_FLIP_IDX}\nnames:\n  0: person\n", encoding="utf-8")
    return p


def build_dataset(items: Sequence[Item], teacher, out_dir: Path = DEFAULT_OUT, val_frac: float = 0.1,
                  seed: int = 0, keep_kwargs: Optional[Dict[str, Any]] = None, log_every: int = 500
                  ) -> Dict[str, Any]:
    """
    항목(영상, 위치) -> 교사 의사 라벨 -> YOLO pose 데이터셋. 반환: 통계 (items.json에도 저장)
    teacher(img) -> 사람 목록 (pseudo.TeacherTTA)
    """
    import cv2

    from training.pose.pseudo import keep_person

    out_dir = Path(out_dir)
    split = split_by_video(items, val_frac, seed)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    by_video: Dict[str, List[Item]] = {}
    for it in items:
        by_video.setdefault(it["video"], []).append(it)
    kept, records = 0, []
    stats = {"frames": 0, "kept": 0, "no_person": 0, "uncertain": 0, "persons": 0, "angle": {}}
    for vi, (video, its) in enumerate(sorted(by_video.items())):
        frames = read_frames(video, [it["frac"] for it in its])
        for it in its:
            img = frames.get(it["frac"])
            if img is None:
                continue
            stats["frames"] += 1
            persons = teacher(img)
            rec = {k: it[k] for k in ("video", "frac", "group", "source", "cls")}
            if not persons:
                stats["no_person"] += 1
                rec["status"] = "no_person"
            elif not all(keep_person(p, **(keep_kwargs or {})) for p in persons):
                stats["uncertain"] += 1
                rec["status"] = "uncertain"
            else:
                h, w = img.shape[:2]
                lines = yolo_label_lines(persons, w, h)
                if not lines:
                    stats["uncertain"] += 1
                    rec["status"] = "uncertain"
                else:
                    part = split[it["group"]]
                    name = f"{it['source']}_{kept:06d}"
                    cv2.imwrite(str(out_dir / "images" / part / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    (out_dir / "labels" / part / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                    kept += 1
                    stats["kept"] += 1
                    stats["persons"] += len(lines)
                    for p in persons:
                        a = str(p["angle"])
                        stats["angle"][a] = stats["angle"].get(a, 0) + 1
                    rec.update(status="kept", split=part, name=name, n_persons=len(lines),
                               angles=[int(p["angle"]) for p in persons])
            records.append(rec)
        if log_every and (vi + 1) % log_every == 0:
            print(f"[pose-data] videos {vi + 1}/{len(by_video)} frames {stats['frames']} kept {stats['kept']}",
                  flush=True)
    stats["by_source_cls"] = _count_kept(records)
    write_data_yaml(out_dir)
    (out_dir / "items.json").write_text(json.dumps({"stats": stats, "records": records}, ensure_ascii=False,
                                                   indent=1), encoding="utf-8")
    return stats


def _count_kept(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for r in records:
        key = f"{r['source']}/{r['cls']}"
        d = out.setdefault(key, {"frames": 0, "kept": 0})
        d["frames"] += 1
        d["kept"] += int(r.get("status") == "kept")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="포즈 파인튜닝용 의사 라벨 데이터셋 만들기")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--manifest", default=str(REPO_ROOT / "dataset" / "_manifest" / "manifest_v1.jsonl"))
    ap.add_argument("--teacher", default=str(REPO_ROOT / "dataset" / "_models" / "yolov8x-pose.pt"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="점검용: 항목 수 상한")
    args = ap.parse_args(argv)

    from training.pose.pseudo import TeacherTTA

    items = plan_wanfall(seed=args.seed) + plan_figshare(Path(args.manifest))
    if args.limit:
        items = random.Random(args.seed).sample(items, min(args.limit, len(items)))
    print(f"[pose-data] {len(items)} frames from {len({it['video'] for it in items})} videos -> {args.out}", flush=True)
    stats = build_dataset(items, TeacherTTA(args.teacher, device=args.device), Path(args.out), seed=args.seed)
    print(json.dumps({k: v for k, v in stats.items() if k != "by_source_cls"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
