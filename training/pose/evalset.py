"""
포즈 평가셋 (사람이 확인하는 키포인트 정답): 평가 데이터셋 영상에서 낙상 뒤 누운 자세 위주로 프레임을 고르고
교사 모델 + 회전 TTA로 초안 라벨을 만들어 CVAT에서 고칠 수 있게 COCO Keypoints 형식으로 내보냅니다.
고친 COCO 파일을 다시 YOLO 형식으로 바꿔 기본/파인튜닝 모델의 OKS AP를 같은 평가기(ultralytics val)로 잽니다.

이 평가셋은 모델 선택(체크포인트, 의사 라벨 임계값)에 쓰지 않습니다. 정답은 사람이 확인한 뒤에만 의미가 있고,
초안 그대로 재면 교사 모델과 비슷한 모델이 유리하게 나옵니다.

명령
  python -m training.pose.evalset draft --out dataset/_pose/eval_v1          # 프레임 + 초안 COCO
  python -m training.pose.evalset to-yolo --coco <고친 annotations.json> --out dataset/_pose/eval_v1
  python -m training.pose.evalset eval --data dataset/_pose/eval_v1/data.yaml --models <base.pt> <ft.pt>
"""
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "dataset" / "_pose" / "eval_v1"
COCO_KP_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                 "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
                 "right_knee", "left_ankle", "right_ankle"]
COCO_SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                 [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
# 데이터셋별 프레임 수 (낙상 클립에서 클립당 최대 1장, 서로 다른 클립)
DEFAULT_QUOTA = {"le2i": 35, "caucafall": 25, "gmdcsa24": 30, "mcfd": 40, "urfd": 20}

__all__ = ["select_frames", "to_coco", "coco_to_yolo", "main"]


def select_frames(rows: Sequence[Dict[str, Any]], quota: Dict[str, int] = DEFAULT_QUOTA, seed: int = 0
                  ) -> List[Dict[str, Any]]:
    """
    낙상 클립에서 프레임 시각 고르기: 첫 낙상 구간 끝 이후 [fall_end + 0.3, fall_end + 2.0] 안의 무작위 시각
    (클립 끝을 넘으면 클립 끝 직전). MCFD는 시나리오(event_group)당 한 카메라만 골라 같은 사건이 몰리지 않게 함.
    """
    rng = random.Random(seed)
    out = []
    for ds, n in quota.items():
        cands = [r for r in rows if r.get("dataset") == ds and r.get("label") == "fall" and r.get("fall_end_s")]
        if ds == "mcfd":
            by_g: Dict[str, List[Dict[str, Any]]] = {}
            for r in cands:
                by_g.setdefault(r["event_group"], []).append(r)
            picked = []
            groups = sorted(by_g)
            rng.shuffle(groups)
            # 시나리오를 돌아가며 카메라 하나씩 (시나리오 24개 < 40장이면 두 번째 바퀴는 다른 카메라)
            used: Dict[str, set] = {g: set() for g in groups}
            while len(picked) < n and any(len(used[g]) < len(by_g[g]) for g in groups):
                for g in groups:
                    left = [r for r in by_g[g] if r["clip_id"] not in used[g]]
                    if left and len(picked) < n:
                        r = rng.choice(left)
                        used[g].add(r["clip_id"])
                        picked.append(r)
        else:
            picked = rng.sample(cands, min(n, len(cands)))
        for r in sorted(picked, key=lambda x: x["clip_id"]):
            fe = float(r["fall_end_s"])
            out.append({"clip_id": r["clip_id"], "dataset": ds, "video": r["video"], "crop_px": r.get("crop_px"),
                        "t": round(fe + rng.uniform(0.3, 2.0), 3)})
    return out


def _read_at(video: Path, t: float, crop_px=None):
    """영상의 t초 프레임 (fps 기준, 클립 끝을 넘으면 마지막 프레임). crop_px가 있으면 자름"""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target = int(round(t * fps))
    if n > 0:
        target = min(target, n - 1)
    img, i = None, 0
    while i <= target:
        ok, frame = cap.read()
        if not ok:
            break
        img = frame
        i += 1
    cap.release()
    if img is not None and crop_px:
        x1, y1, x2, y2 = (int(v) for v in crop_px)
        img = img[y1:y2, x1:x2]
    return img


def to_coco(images: List[Dict[str, Any]], persons_per_image: List[List[Dict[str, Any]]],
            min_conf: float = 0.3) -> Dict[str, Any]:
    """이미지 목록과 사람 목록(원본 픽셀) -> COCO Keypoints dict (v = 2 if conf >= 0.5 else 1 if >= 0.2 else 0)"""
    anns, aid = [], 1
    for img, persons in zip(images, persons_per_image):
        for p in persons:
            if float(p["conf"]) < min_conf:
                continue
            x1, y1, x2, y2 = (float(v) for v in p["box"])
            kps, nk = [], 0
            for x, y, c in np.asarray(p["kps"], dtype=np.float64):
                v = 2 if c >= 0.5 else (1 if c >= 0.2 else 0)
                kps += [round(float(x), 1), round(float(y), 1), v] if v else [0, 0, 0]
                nk += int(v > 0)
            anns.append({"id": aid, "image_id": img["id"], "category_id": 1, "iscrowd": 0,
                         "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                         "area": round((x2 - x1) * (y2 - y1), 1), "keypoints": kps, "num_keypoints": nk,
                         "score": round(float(p["conf"]), 3)})
            aid += 1
    return {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "person", "supercategory": "person",
                            "keypoints": COCO_KP_NAMES, "skeleton": COCO_SKELETON}]}


def coco_to_yolo(coco: Dict[str, Any], out_dir: Path, images_dir: Path) -> int:
    """고친 COCO -> out_dir/labels/val/*.txt (images/val은 images_dir의 파일을 씀). 반환: 라벨 파일 수"""
    out_dir = Path(out_dir)
    lab_dir = out_dir / "labels" / "val"
    lab_dir.mkdir(parents=True, exist_ok=True)
    by_img: Dict[int, List[Dict[str, Any]]] = {}
    for a in coco.get("annotations", []):
        by_img.setdefault(int(a["image_id"]), []).append(a)
    n = 0
    for im in coco["images"]:
        w, h = float(im["width"]), float(im["height"])
        lines = []
        for a in by_img.get(int(im["id"]), []):
            x, y, bw, bh = (float(v) for v in a["bbox"])
            vals = [(x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h]
            k = a.get("keypoints") or [0] * 51
            for j in range(17):
                kx, ky, kv = float(k[3 * j]), float(k[3 * j + 1]), int(k[3 * j + 2])
                vals += [kx / w, ky / h, 2] if kv > 0 else [0.0, 0.0, 0]
            lines.append("0 " + " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in vals))
        (lab_dir / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""),
                                                                       encoding="utf-8")
        n += 1
    from training.pose.dataset import COCO_FLIP_IDX
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve().as_posix()}\ntrain: images/val\nval: images/val\nkpt_shape: [17, 3]\n"
        f"flip_idx: {COCO_FLIP_IDX}\nnames:\n  0: person\n", encoding="utf-8")
    return n


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="포즈 평가셋: 초안 만들기 / YOLO 변환 / 평가")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft")
    d.add_argument("--out", default=str(DEFAULT_OUT))
    d.add_argument("--manifest", default=str(REPO_ROOT / "dataset" / "_manifest" / "manifest_v1.jsonl"))
    d.add_argument("--teacher", default=str(REPO_ROOT / "dataset" / "_models" / "yolov8x-pose.pt"))
    d.add_argument("--device", default="0")
    d.add_argument("--seed", type=int, default=0)
    y = sub.add_parser("to-yolo")
    y.add_argument("--coco", required=True)
    y.add_argument("--out", default=str(DEFAULT_OUT))
    e = sub.add_parser("eval")
    e.add_argument("--data", default=str(DEFAULT_OUT / "data.yaml"))
    e.add_argument("--models", nargs="+", required=True)
    e.add_argument("--device", default="0")
    args = ap.parse_args(argv)

    if args.cmd == "draft":
        import cv2

        from training.pose.pseudo import TeacherTTA
        out = Path(args.out)
        (out / "images" / "val").mkdir(parents=True, exist_ok=True)
        rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        frames = select_frames(rows, seed=args.seed)
        teacher = TeacherTTA(args.teacher, device=args.device)
        images, persons = [], []
        for i, f in enumerate(frames):
            img = _read_at(REPO_ROOT / f["video"], f["t"], f.get("crop_px"))
            if img is None:
                continue
            name = f"{i:04d}_{f['clip_id'].replace('/', '__')}.jpg"
            cv2.imwrite(str(out / "images" / "val" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            h, w = img.shape[:2]
            images.append({"id": len(images) + 1, "file_name": name, "width": w, "height": h,
                           "clip_id": f["clip_id"], "dataset": f["dataset"], "t": f["t"]})
            persons.append(teacher(img))
        coco = to_coco(images, persons)
        (out / "annotations_draft.json").write_text(json.dumps(coco, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "frames.json").write_text(json.dumps(frames, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[pose-eval] {len(images)} images, {len(coco['annotations'])} draft persons -> {out}")
        return 0
    if args.cmd == "to-yolo":
        coco = json.loads(Path(args.coco).read_text(encoding="utf-8"))
        n = coco_to_yolo(coco, Path(args.out), Path(args.out) / "images" / "val")
        print(f"[pose-eval] {n} label files -> {Path(args.out) / 'labels' / 'val'}")
        return 0
    from ultralytics import YOLO
    res = {}
    for m in args.models:
        r = YOLO(m).val(data=args.data, split="val", device=args.device, imgsz=640, batch=8, verbose=False,
                        plots=False)
        res[m] = {"pose_mAP50_95": float(r.pose.map), "pose_mAP50": float(r.pose.map50),
                  "box_mAP50_95": float(r.box.map)}
        print(m, res[m])
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
