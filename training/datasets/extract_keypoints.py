"""
매니페스트의 영상마다 YOLO Pose 키포인트를 뽑아 클립 파일(clipfmt, .npz)로 저장

  python -m training.datasets.extract_keypoints --manifest dataset/_manifest/manifest_v1.jsonl
  python -m training.datasets.extract_keypoints --manifest ... --datasets le2i caucafall --limit 20 --device 0

- 운영과 같은 모델(training.paths.YOLO_POSE_MODEL, dataset/_models/ 아래), 같은 신뢰도(app.config.DEFAULT_CONF),
  같은 대상 선택(app.engine.target_select: 이전 프레임 박스와 IoU로 대상 유지)을 사용
- 시간축은 프레임 번호 / 컨테이너 FPS (FPS를 못 읽으면 30으로 두고 meta.fps_source="fallback")
- 출력: <out>/<clip_id>.npz. 이미 있으면 건너뜀(이어하기). 결과는 <out>/_extract_log.jsonl에 한 줄씩 기록
- 영상 하나가 실패해도 전체 실행은 계속됨
- 원본 영상과 추출 결과는 데이터셋 라이선스상 git에 올리지 않음 (dataset/ 전체가 git 제외)

--limit N 은 이번 실행에서 새로 추출을 시도할 클립 수입니다 (이미 끝난 클립은 세지 않음).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from app.engine.target_select import select_target_index
from training.datasets.clipfmt import CLIP_FORMAT_VERSION, NUM_KPS, save_clip
from training.paths import REPO_ROOT, YOLO_POSE_MODEL

MODELS_DIR = REPO_ROOT / "dataset" / "_models"
DEFAULT_OUT = REPO_ROOT / "dataset" / "_keypoints" / "v1"
LOG_NAME = "_extract_log.jsonl"
FALLBACK_FPS = 30.0
IMGSZ = 640
IOU_THRESH = 0.3
SELECT_NAME = f"target_select(iou>={IOU_THRESH}, else top-conf)"


# 프레임 하나의 탐지 결과: (boxes_xyxy (N,4) 픽셀, confs (N,), kp_xy (N,17,2) 픽셀, kp_conf (N,17))
Detections = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# 프레임 묶음 -> 프레임별 탐지 결과
Predictor = Callable[[List[np.ndarray]], List[Detections]]
# 영상 경로 -> (컨테이너 FPS 또는 None, BGR 프레임 이터러블)
VideoOpener = Callable[[Path], Tuple[Optional[float], Iterable[np.ndarray]]]


def empty_detections() -> Detections:
    return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
            np.zeros((0, NUM_KPS, 2), np.float32), np.zeros((0, NUM_KPS), np.float32))


# ---------------------------------------------------------------------------
# 핵심: 프레임별 탐지 -> 클립 배열 (cv2/ultralytics 없이 테스트 가능)
# ---------------------------------------------------------------------------

def extract_frames(frames: Iterable[np.ndarray], fps: float, predictor: Predictor,
                   batch: int = 8, iou_thresh: float = IOU_THRESH) -> Dict[str, Any]:
    """
    프레임을 순서대로 추론하고 런타임과 같은 방식으로 대상 1명을 골라 클립 배열을 만듭니다.

    - 직전 프레임에서 고른 박스(픽셀)를 last_bbox로 넘겨 IoU 추적 (사람이 없으면 None으로 리셋, 런타임과 같음)
    - 좌표는 해당 프레임 크기로 정규화하고 [0, 1]로 자름 (obs_schema와 같음)
    - switched (T,): 그 프레임에서 대상이 바뀌었는지 (런타임 obs.target_switched와 같음: 직전 프레임에 대상이 있었고
      IoU >= iou_thresh인 후보가 없어 최고 신뢰도 후보로 바꾼 경우만 True. 놓친 뒤 다시 잡은 첫 프레임은 False)
    - n_switch = switched.sum()
    """
    ts: List[float] = []
    kps: List[np.ndarray] = []
    bbox: List[np.ndarray] = []
    present: List[bool] = []
    switched_l: List[bool] = []
    frame_wh: Optional[List[int]] = None
    last_bbox = None
    n_multi = 0

    def consume(buf: List[np.ndarray]):
        nonlocal last_bbox, n_multi, frame_wh
        dets = predictor(buf)
        if len(dets) != len(buf):
            raise RuntimeError(f"predictor returned {len(dets)} results for {len(buf)} frames")
        for frame, (boxes, confs, kxy, kcf) in zip(buf, dets):
            h, w = frame.shape[:2]
            if frame_wh is None:
                frame_wh = [int(w), int(h)]
            ts.append(len(ts) / fps)
            if len(boxes) > 1:
                n_multi += 1
            idx, switched = select_target_index(boxes, confs, last_bbox=last_bbox, iou_thresh=iou_thresh)
            switched_l.append(bool(switched) and idx is not None)
            if idx is None:
                last_bbox = None
                kps.append(np.zeros((NUM_KPS, 3), np.float32))
                bbox.append(np.zeros(4, np.float32))
                present.append(False)
                continue
            last_bbox = boxes[idx]
            k = np.zeros((NUM_KPS, 3), np.float32)
            k[:, 0] = np.clip(np.asarray(kxy[idx][:, 0], np.float64) / w, 0.0, 1.0)
            k[:, 1] = np.clip(np.asarray(kxy[idx][:, 1], np.float64) / h, 0.0, 1.0)
            k[:, 2] = np.asarray(kcf[idx], np.float32)
            b = np.asarray(boxes[idx], np.float64)
            kps.append(k)
            bbox.append(np.clip(b / np.array([w, h, w, h], np.float64), 0.0, 1.0).astype(np.float32))
            present.append(True)

    buf: List[np.ndarray] = []
    for frame in frames:
        buf.append(frame)
        if len(buf) >= max(1, int(batch)):
            consume(buf)
            buf = []
    if buf:
        consume(buf)

    t = len(ts)
    sw = np.asarray(switched_l, bool)
    return {
        "ts": np.asarray(ts, np.float64),
        "kps": np.stack(kps).astype(np.float32) if t else np.zeros((0, NUM_KPS, 3), np.float32),
        "bbox": np.stack(bbox).astype(np.float32) if t else np.zeros((0, 4), np.float32),
        "present": np.asarray(present, bool),
        "frame_wh": frame_wh or [0, 0],
        "switched": sw,
        "n_switch": int(sw.sum()),
        "n_multi": n_multi,
    }


def normalize_fps(fps: Optional[float]) -> Tuple[float, str]:
    """컨테이너 FPS가 비정상이면 30으로 대체합니다."""
    try:
        f = float(fps) if fps is not None else 0.0
    except (TypeError, ValueError):
        f = 0.0
    if np.isfinite(f) and 1.0 <= f <= 240.0:
        return f, "container"
    return FALLBACK_FPS, "fallback"


def clip_out_path(out_dir: Path, clip_id: str) -> Path:
    """clip_id -> <out>/<clip_id>.npz (경로 이탈과 Windows 금지 문자를 막음)."""
    if not isinstance(clip_id, str) or not clip_id.strip():
        raise ValueError(f"invalid clip_id: {clip_id!r}")
    parts = [p for p in clip_id.replace("\\", "/").split("/") if p]
    if not parts or any(p in (".", "..") for p in parts):
        raise ValueError(f"invalid clip_id: {clip_id!r}")
    bad = '<>:"|?*'
    parts = ["".join("_" if c in bad else c for c in p) for p in parts]
    return out_dir.joinpath(*parts[:-1], parts[-1] + ".npz")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _append_log(log_path: Path, rec: Dict[str, Any]):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def extract_one(row: Dict[str, Any], out_path: Path, predictor: Predictor, open_video: VideoOpener,
                extractor_meta: Dict[str, Any], repo_root: Path = REPO_ROOT, batch: int = 8,
                with_hash: bool = False) -> Dict[str, Any]:
    """매니페스트 한 줄을 추출해 저장하고 통계를 반환합니다. 실패하면 예외."""
    video = repo_root / row["video"]
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {row['video']}")
    raw_fps, frames = open_video(video)
    container_fps, fps_source = normalize_fps(raw_fps)
    fps = container_fps
    if row.get("fps") is not None:
        # manifest가 지정한 실제 촬영 fps (헤더 fps가 틀린 데이터셋)
        fps, fps_source = float(row["fps"]), "manifest"
    container_frames = _container_frame_count(frames)
    crop = row.get("crop_px")
    if crop is not None:
        frames = CroppedFrames(frames, crop)
    clip = extract_frames(frames, fps, predictor, batch=batch)
    t = int(clip["ts"].shape[0])
    if t == 0:
        raise RuntimeError("no frames decoded")
    check_label_timebase(row, t, fps)
    present_ratio = float(clip["present"].mean())
    meta = {
        "clip_format_version": CLIP_FORMAT_VERSION,
        "dataset": row.get("dataset"),
        "clip_id": row["clip_id"],
        "video": row["video"],
        "native_fps": fps,
        "fps_source": fps_source,
        "container_fps": container_fps,
        "n_frames": t,
        "extractor": dict(extractor_meta),
        "present_ratio": round(present_ratio, 4),
        "n_target_switch": clip["n_switch"],
        "n_multi_person_frames": clip["n_multi"],
        # 컨테이너가 알려준 프레임 수 (디코딩이 중간에 끊겨 클립이 짧아졌는지 확인용, 모르면 None)
        "container_frames": container_frames,
        # 원본 영상에서 잘라 쓴 영역 (kps/bbox/frame_wh는 잘라낸 영역 기준)
        "crop_px": list(crop) if crop is not None else None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if with_hash:
        meta["video_sha256"] = sha256_file(video)
    save_clip(out_path, clip["ts"], clip["kps"], clip["bbox"], clip["present"], clip["frame_wh"], meta,
              switched=clip["switched"])
    return {"frames": t, "container_frames": container_frames, "native_fps": fps, "fps_source": fps_source,
            "present_ratio": round(present_ratio, 4), "switches": clip["n_switch"]}


class CroppedFrames:
    """프레임 이터러블에서 [x1, y1, x2, y2] 영역만 잘라 돌려줍니다. 영역이 프레임 밖이면 ValueError."""

    def __init__(self, frames: Iterable[np.ndarray], crop: Sequence[int]):
        self.frames = frames
        self.crop = [int(v) for v in crop]
        self.frame_count = getattr(frames, "frame_count", None)

    def __iter__(self):
        x1, y1, x2, y2 = self.crop
        for f in self.frames:
            h, w = f.shape[:2]
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError(f"crop_px {self.crop} outside frame {w}x{h}")
            yield f[y1:y2, x1:x2]


def check_label_timebase(row: Dict[str, Any], n_frames: int, fps: float) -> None:
    """
    라벨 구간이 클립 길이를 크게 넘으면 시간축이 어긋난 것 (예: 헤더 fps 오류) -> ValueError로 이 클립만 실패.
    허용: 길이 x 1.05 + 0.5초 (le2i 일부 라벨이 끝을 0.6초 넘음)
    """
    segs = row.get("segments") or []
    if not segs:
        return
    duration = n_frames / fps
    last = max(float(s["end_s"]) for s in segs)
    if last > duration * 1.05 + 0.5:
        raise ValueError(f"label/timebase mismatch: last segment ends at {last:.2f}s but clip is "
                         f"{duration:.2f}s ({n_frames} frames @ {fps:g}fps); set manifest fps")


def _container_frame_count(frames: Any) -> Optional[int]:
    """open_video가 돌려준 프레임 이터러블의 frame_count 속성 (없거나 비정상이면 None)."""
    try:
        n = int(getattr(frames, "frame_count", None))
    except (TypeError, ValueError, OverflowError):
        return None
    return n if n > 0 else None


def run_extraction(rows: Sequence[Dict[str, Any]], out_dir: Path, get_predictor: Callable[[], Predictor],
                   open_video: VideoOpener, extractor_meta: Dict[str, Any], limit: Optional[int] = None,
                   repo_root: Path = REPO_ROOT, batch: int = 8, with_hash: bool = False,
                   log=print) -> Dict[str, int]:
    """
    여러 클립을 추출합니다. 모델은 실제로 추출할 클립이 있을 때만 get_predictor()로 한 번 만듭니다.
    클립 하나의 실패는 로그에 남기고 다음으로 넘어갑니다.
    """
    out_dir = Path(out_dir)
    log_path = out_dir / LOG_NAME
    stats = {"total": len(rows), "done": 0, "failed": 0, "skipped": 0}
    predictor: Optional[Predictor] = None
    attempted = 0
    for row in rows:
        clip_id = row.get("clip_id")
        try:
            out_path = clip_out_path(out_dir, clip_id)
        except ValueError as e:
            stats["failed"] += 1
            _append_log(log_path, {"clip_id": clip_id, "ok": False, "frames": 0, "secs": 0.0, "error": str(e)})
            continue
        if out_path.exists():
            stats["skipped"] += 1
            continue
        if limit is not None and attempted >= limit:
            break
        attempted += 1
        t0 = time.time()
        rec: Dict[str, Any] = {"clip_id": clip_id, "ok": False, "frames": 0, "secs": 0.0, "error": None}
        try:
            if not (repo_root / str(row.get("video") or "")).is_file():
                raise FileNotFoundError(f"video not found: {row.get('video')}")  # 모델 로드 전에 확인
            if predictor is None:
                predictor = get_predictor()
            info = extract_one(row, out_path, predictor, open_video, extractor_meta,
                               repo_root=repo_root, batch=batch, with_hash=with_hash)
            rec.update(ok=True, **info)
            stats["done"] += 1
        except Exception as e:  # 영상 하나 때문에 전체가 멈추지 않도록
            rec["error"] = f"{type(e).__name__}: {e}"
            stats["failed"] += 1
        rec["secs"] = round(time.time() - t0, 2)
        _append_log(log_path, rec)
        status = "ok" if rec["ok"] else f"FAIL {rec['error']}"
        log(f"[{attempted}] {clip_id}: {status} frames={rec.get('frames', 0)} "
            f"present={rec.get('present_ratio', '-')} {rec['secs']}s")
    return stats


# ---------------------------------------------------------------------------
# 실제 의존성 (cv2 / ultralytics) - CLI에서만 import
# ---------------------------------------------------------------------------

def open_video_cv2(path: Path) -> Tuple[Optional[float], Iterable[np.ndarray]]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps, _Cv2Frames(cap, cap.get(cv2.CAP_PROP_FRAME_COUNT))


class _Cv2Frames:
    """cv2 프레임 이터러블. frame_count(컨테이너 기록값)로 디코딩이 일찍 끝났는지 비교할 수 있게 함."""

    def __init__(self, cap, frame_count):
        self.cap = cap
        self.frame_count = frame_count

    def __iter__(self):
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    break
                yield frame
        finally:
            self.cap.release()


def resolve_model_path(model: Optional[str]) -> Path:
    """
    모델 경로를 정합니다. 지정하지 않으면 dataset/_models/<YOLO_POSE_MODEL>.
    파일 이름만 주면 dataset/_models/ 아래에서 찾고, 없으면 ultralytics 공개 에셋에서 그 위치로 받습니다.
    """
    p = Path(model) if model else MODELS_DIR / YOLO_POSE_MODEL
    if p.is_file():
        return p
    if not p.is_absolute() and len(p.parts) == 1:
        p = MODELS_DIR / p.name
        if p.is_file():
            return p
    p.parent.mkdir(parents=True, exist_ok=True)
    from ultralytics.utils.downloads import attempt_download_asset

    got = Path(attempt_download_asset(str(p)))
    if not got.is_file():
        raise FileNotFoundError(f"YOLO pose model not found: {p}")
    return got


def make_yolo_predictor(model_path: Path, device: Optional[str], conf: float, imgsz: int = IMGSZ,
                        half: bool = False) -> Predictor:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    dev: Any = device if device is not None else ("0" if _cuda_available() else "cpu")
    if isinstance(dev, str) and dev.isdigit():
        dev = int(dev)  # 런타임과 같이 device=0

    def predict(frames: List[np.ndarray]) -> List[Detections]:
        kw: Dict[str, Any] = {"half": True} if half else {}  # 최신 ultralytics는 half 인자 자체에 경고를 냄
        results = model.predict(frames, device=dev, conf=conf, imgsz=imgsz, verbose=False, **kw)
        out: List[Detections] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                out.append(empty_detections())
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            n = len(boxes)
            if r.keypoints is not None and len(r.keypoints) == n:
                kxy = r.keypoints.xy.cpu().numpy()
                kcf = (r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None
                       else np.ones((n, kxy.shape[1]), np.float32))
            else:
                kxy = np.zeros((n, NUM_KPS, 2), np.float32)
                kcf = np.zeros((n, NUM_KPS), np.float32)
            out.append((boxes, confs, kxy, kcf))
        return out

    return predict


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    """매니페스트 JSONL을 읽습니다 (manifest.load_manifest가 있으면 그것을 사용)."""
    try:
        from training.datasets.manifest import load_manifest
    except ImportError:
        load_manifest = None
    if load_manifest is not None:
        return list(load_manifest(path))
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="매니페스트 영상 -> YOLO Pose 키포인트 클립(.npz)")
    ap.add_argument("--manifest", required=True, help="통합 매니페스트 JSONL")
    ap.add_argument("--datasets", nargs="*", default=None, help="이 데이터셋만 (예: le2i caucafall)")
    ap.add_argument("--limit", type=int, default=None, help="이번 실행에서 새로 추출할 최대 클립 수")
    ap.add_argument("--device", default=None, help="cpu 또는 GPU 번호 (기본: CUDA 있으면 0)")
    ap.add_argument("--model", default=None, help=f"YOLO pose 가중치 (기본: dataset/_models/{YOLO_POSE_MODEL})")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="출력 폴더")
    ap.add_argument("--batch", type=int, default=8, help="추론 배치 크기 (결과에는 영향 없음)")
    ap.add_argument("--half", action="store_true", help="FP16 추론 (GPU 전용)")
    ap.add_argument("--hash", action="store_true", help="원본 영상 sha256을 meta에 기록")
    args = ap.parse_args(argv)

    from app.config import DEFAULT_CONF

    rows = read_manifest(Path(args.manifest))
    if args.datasets:
        wanted = set(args.datasets)
        rows = [r for r in rows if r.get("dataset") in wanted]
    out_dir = Path(args.out)

    try:
        import ultralytics
        ul_version = getattr(ultralytics, "__version__", None)
    except ImportError:
        ul_version = None
    model_name = Path(args.model).name if args.model else YOLO_POSE_MODEL
    extractor_meta = {
        "model": model_name,
        "conf": float(DEFAULT_CONF),
        "imgsz": IMGSZ,
        "select": SELECT_NAME,
        "ultralytics_version": ul_version,
        "half": bool(args.half),
        "device": args.device,
    }

    def get_predictor() -> Predictor:
        path = resolve_model_path(args.model)
        print(f"[extract] model={path} conf={DEFAULT_CONF} imgsz={IMGSZ} device={args.device or 'auto'}")
        return make_yolo_predictor(path, args.device, float(DEFAULT_CONF), IMGSZ, half=args.half)

    print(f"[extract] {len(rows)} clips -> {out_dir}")
    stats = run_extraction(rows, out_dir, get_predictor, open_video_cv2, extractor_meta, limit=args.limit,
                           batch=args.batch, with_hash=args.hash)
    print(f"[extract] done={stats['done']} failed={stats['failed']} skipped(existing)={stats['skipped']} "
          f"total={stats['total']} log={out_dir / LOG_NAME}")
    return 1 if stats["failed"] and not stats["done"] else 0


if __name__ == "__main__":
    sys.exit(main())
