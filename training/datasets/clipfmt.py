"""
영상 1개 = 키포인트 클립 파일 1개 (.npz, numpy만 사용)

공개 데이터셋 추출(extract_keypoints), 운영 녹화(obs JSONL), 이벤트 sidecar를 같은 형식으로 맞춰
이후 리샘플링/특징(features v2)/학습이 출처와 무관하게 동작하도록 합니다.

키 (np.savez_compressed)
  ts        float64 (T,)       영상 시작 기준 초
  kps       float32 (T,17,3)   COCO-17 [x/frame_w, y/frame_h, conf] (obs_schema/kp_buffer와 같은 정규화), 사람 없음 -> 0
  bbox      float32 (T,4)      [x1,y1,x2,y2] 정규화, 사람 없음 -> 0
  present   bool    (T,)       해당 프레임에서 대상이 선택되었는지
  frame_wh  int32   (2,)       원본 [width, height] (픽셀)
  switched  bool    (T,)       (선택) 해당 프레임에서 추적 대상이 바뀌었는지 (런타임 obs.target_switched와 같은 정의).
                               예전 파일/sidecar 클립에는 없음 -> load_clip 결과에서 키 자체가 빠짐 (clip.get("switched") is None)
  n_cand    int16   (T,)       (선택) 프레임에서 검출된 사람 후보 수 (대상 전환 정책 비교용, 예전 파일에는 없음)
  fallback  bool    (T,)       (선택) 원래 방향에서 아무도 못 찾아 회전 폴백(app.engine.rotate_fallback)으로 찾은 프레임
  switched_base bool (T,)      (선택) 회전 폴백을 쓰지 않았을 때의 대상 전환 (폴백 끈 재생용)
  meta      0-d str            JSON (clip_format_version, dataset, clip_id, video, native_fps, n_frames,
                               extractor{model,conf,imgsz,select,ultralytics_version}, video_sha256?, created_at)

switched는 없어도 되는 추가 키라 clip_format_version은 1 그대로 둡니다 (예전 파일도 그대로 읽히고,
예전 load_clip도 모르는 키를 무시하므로 새 파일을 읽을 수 있음).
"""
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

CLIP_FORMAT_VERSION = 1
NUM_KPS = 17
CLIP_KEYS = ("ts", "kps", "bbox", "present", "frame_wh", "meta")
OPTIONAL_KEYS = ("switched", "n_cand", "fallback", "switched_base")
# bool (T,) 선택 키: fallback = 회전 폴백으로 찾은 프레임, switched_base = 폴백을 쓰지 않았을 때의 대상 전환
BOOL_OPTIONAL_KEYS = ("switched", "fallback", "switched_base")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _as_switched(switched, t: int, name: str = "switched") -> np.ndarray:
    """bool 선택 키 -> bool (T,). 길이가 다르거나 0/1이 아닌 값이 있으면 ValueError."""
    arr = np.asarray(switched)
    if arr.dtype != np.bool_:
        if arr.dtype.kind not in "iuf" or (arr.size and not np.all((arr == 0) | (arr == 1))):
            raise ValueError(f"{name} must be bool (or 0/1)")
    arr = arr.astype(bool)
    if arr.shape != (t,):
        raise ValueError(f"{name} shape {arr.shape} != ({t},)")
    return arr


def _as_n_cand(n_cand, t: int) -> np.ndarray:
    """n_cand -> int16 (T,) 0 이상. 길이가 다르거나 음수/정수가 아닌 값이 있으면 ValueError."""
    arr = np.asarray(n_cand)
    if arr.dtype.kind not in "iu" and not (arr.dtype.kind == "f" and np.all(np.mod(arr, 1) == 0)):
        raise ValueError("n_cand must be integers")
    arr = arr.astype(np.int64)
    if arr.shape != (t,):
        raise ValueError(f"n_cand shape {arr.shape} != ({t},)")
    if np.any(arr < 0):
        raise ValueError("n_cand must be >= 0")
    return arr.astype(np.int16)


def _as_arrays(ts, kps, bbox, present, frame_wh, switched=None, n_cand=None, fallback=None,
               switched_base=None) -> Dict[str, np.ndarray]:
    """형식에 맞는 dtype으로 바꾸고 shape를 검사합니다. 어긋나면 ValueError. switched가 None이면 키를 넣지 않음."""
    ts = np.asarray(ts, dtype=np.float64).reshape(-1)
    t = ts.shape[0]
    kps = np.asarray(kps, dtype=np.float32)
    if t == 0 and kps.size == 0:
        kps = kps.reshape(0, NUM_KPS, 3)
    bbox = np.asarray(bbox, dtype=np.float32)
    if t == 0 and bbox.size == 0:
        bbox = bbox.reshape(0, 4)
    present = np.asarray(present, dtype=bool).reshape(-1)
    frame_wh = np.asarray(frame_wh, dtype=np.int32).reshape(-1)

    if kps.shape != (t, NUM_KPS, 3):
        raise ValueError(f"kps shape {kps.shape} != ({t}, {NUM_KPS}, 3)")
    if bbox.shape != (t, 4):
        raise ValueError(f"bbox shape {bbox.shape} != ({t}, 4)")
    if present.shape != (t,):
        raise ValueError(f"present shape {present.shape} != ({t},)")
    if frame_wh.shape != (2,):
        raise ValueError(f"frame_wh shape {frame_wh.shape} != (2,)")
    if not np.all(np.isfinite(ts)):
        raise ValueError("ts has non-finite values")
    if t >= 2 and np.any(np.diff(ts) < 0):
        raise ValueError("ts must be non-decreasing")
    if not (np.all(np.isfinite(kps)) and np.all(np.isfinite(bbox))):
        raise ValueError("kps/bbox have non-finite values")
    out = {"ts": ts, "kps": kps, "bbox": bbox, "present": present, "frame_wh": frame_wh}
    if switched is not None:
        out["switched"] = _as_switched(switched, t)
    if n_cand is not None:
        out["n_cand"] = _as_n_cand(n_cand, t)
    if fallback is not None:
        out["fallback"] = _as_switched(fallback, t, "fallback")
    if switched_base is not None:
        out["switched_base"] = _as_switched(switched_base, t, "switched_base")
    return out


def save_clip(path, ts, kps, bbox, present, frame_wh, meta: Dict[str, Any], switched=None, n_cand=None,
              fallback=None, switched_base=None) -> None:
    """
    클립 파일을 저장합니다 (검증 후 임시 파일에 쓰고 교체하므로 중단돼도 반쯤 쓴 파일이 남지 않음).
    switched (T,) bool, n_cand (T,) int(프레임의 사람 후보 수), fallback / switched_base (T,) bool은 선택이며
    None이면 파일에 쓰지 않습니다.
    """
    arrays = _as_arrays(ts, kps, bbox, present, frame_wh, switched, n_cand, fallback, switched_base)
    meta = dict(meta or {})
    # 파생 값은 항상 실제 배열 기준으로 덮어씀 (잘라낸 클립을 다시 저장할 때 예전 값이 남지 않도록)
    meta["clip_format_version"] = CLIP_FORMAT_VERSION
    meta["n_frames"] = int(arrays["ts"].shape[0])
    meta.setdefault("created_at", _now_iso())
    meta_json = json.dumps(meta, ensure_ascii=False, default=str)

    path = os.fspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    try:
        # 파일 객체로 넘겨야 numpy가 이름 뒤에 .npz를 붙이지 않음
        with open(tmp, "wb") as f:
            np.savez_compressed(f, meta=np.array(meta_json), **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_clip(path) -> Dict[str, Any]:
    """클립 파일을 읽습니다. meta는 dict로 풀어서 반환합니다. switched는 파일에 있을 때만 키가 생깁니다."""
    with np.load(os.fspath(path), allow_pickle=False) as z:
        missing = [k for k in CLIP_KEYS if k not in z.files]
        if missing:
            raise ValueError(f"clip file missing keys: {missing}")
        out = {k: z[k] for k in CLIP_KEYS if k != "meta"}
        for k in OPTIONAL_KEYS:
            if k in z.files:
                out[k] = z[k]
        meta = json.loads(str(z["meta"][()]))
    ver = int(meta.get("clip_format_version", 0))
    if ver > CLIP_FORMAT_VERSION:
        raise ValueError(f"clip_format_version {ver} is newer than supported {CLIP_FORMAT_VERSION}")
    checked = _as_arrays(out["ts"], out["kps"], out["bbox"], out["present"], out["frame_wh"], out.get("switched"),
                         out.get("n_cand"), out.get("fallback"), out.get("switched_base"))
    for k in OPTIONAL_KEYS:
        if k in checked:
            out[k] = checked[k]
    out["meta"] = meta
    return out


# ---------------------------------------------------------------------------
# 운영 데이터 -> 클립 (obs JSONL 녹화, 이벤트 sidecar)
# ---------------------------------------------------------------------------

def _frame_from_kp_list(kps_list: Optional[Sequence[Dict[str, Any]]]) -> Optional[np.ndarray]:
    """obs keypoints 목록 [{"id","x","y","conf"}, ...] -> (17,3). 비어 있으면 None (kp_buffer.kps_to_frame과 같음)."""
    if not kps_list:
        return None
    frame = np.zeros((NUM_KPS, 3), dtype=np.float32)
    for k in kps_list:
        kid = int(k.get("id", -1))
        if 0 <= kid < NUM_KPS:
            frame[kid] = (float(k.get("x", 0.0)), float(k.get("y", 0.0)), float(k.get("conf", 0.0)))
    return frame


def _frame_from_rows(rows: Optional[Sequence[Sequence[float]]]) -> Optional[np.ndarray]:
    """kp_buffer 프레임 17 x [x, y, conf] -> (17,3). None이면 None."""
    if rows is None:
        return None
    arr = np.asarray(rows, dtype=np.float32)
    if arr.shape != (NUM_KPS, 3):
        return None
    return arr


def _bbox_or_zeros(b) -> np.ndarray:
    if b is None or len(b) != 4:
        return np.zeros(4, dtype=np.float32)
    return np.asarray([float(v) for v in b], dtype=np.float32)


def _default_frame_wh() -> List[int]:
    """해상도 정보가 없는 오래된 기록용 기본값 (런타임 카메라 설정)."""
    try:
        from app.config import FRAME_W, FRAME_H
        return [int(FRAME_W), int(FRAME_H)]
    except Exception:
        return [1920, 1080]


def _assemble(ts_abs: List[float], frames: List[Optional[np.ndarray]], bboxes: List[Any],
              frame_wh: Optional[Sequence[int]], meta: Dict[str, Any],
              switched: Optional[List[bool]] = None) -> Dict[str, Any]:
    t = len(ts_abs)
    kps = np.zeros((t, NUM_KPS, 3), dtype=np.float32)
    bbox = np.zeros((t, 4), dtype=np.float32)
    present = np.zeros(t, dtype=bool)
    for i, (f, b) in enumerate(zip(frames, bboxes)):
        if f is None:
            continue  # missing -> 0, present False
        kps[i] = f
        bbox[i] = _bbox_or_zeros(b)
        present[i] = True
    t0 = float(ts_abs[0]) if t else 0.0
    ts = np.asarray([float(x) - t0 for x in ts_abs], dtype=np.float64)

    meta = dict(meta)
    if frame_wh is None:
        frame_wh = _default_frame_wh()
        meta["frame_wh_source"] = "config_default"
    else:
        meta.setdefault("frame_wh_source", "recorded")
    meta.setdefault("clip_format_version", CLIP_FORMAT_VERSION)
    meta["n_frames"] = t
    meta["t0_abs"] = t0
    if t >= 2 and ts[-1] > 0:
        meta.setdefault("native_fps", round((t - 1) / float(ts[-1]), 3))
    else:
        meta.setdefault("native_fps", None)
    meta.setdefault("created_at", _now_iso())

    arrays = _as_arrays(ts, kps, bbox, present, frame_wh, switched)
    arrays["meta"] = meta
    return arrays


def from_obs_records(obs_list: Iterable[Dict[str, Any]], **meta: Any) -> Dict[str, Any]:
    """
    app.controller 녹화(obs JSONL) 레코드 목록 -> 클립 dict (load_clip과 같은 키).

    - tracks[0].keypoints_raw가 있으면 그것을, 없으면 keypoints를 사용 (스무딩 전 원시값 우선)
    - ts는 첫 obs 기준 상대 시간, frame_shape (h, w)에서 frame_wh를 얻음 (마지막 기록값)
    - 사람이 없는 obs는 kps/bbox 0, present False
    - obs["target_switched"] -> switched (없으면 False)
    - 추가 키워드 인자는 meta에 들어감 (dataset, clip_id 등)
    """
    ts_abs: List[float] = []
    frames: List[Optional[np.ndarray]] = []
    bboxes: List[Any] = []
    switched: List[bool] = []
    frame_wh = None
    for obs in obs_list:
        if not isinstance(obs, dict) or obs.get("ts") is None:
            continue
        tracks = obs.get("tracks") or []
        t0 = tracks[0] if tracks and isinstance(tracks[0], dict) else {}
        frame = _frame_from_kp_list(t0.get("keypoints_raw") or t0.get("keypoints"))
        if frame is not None and t0.get("has_person") is False:
            frame = None
        shape = t0.get("frame_shape")
        if shape and len(shape) == 2 and int(shape[0]) > 0 and int(shape[1]) > 0:
            frame_wh = [int(shape[1]), int(shape[0])]
        ts_abs.append(float(obs["ts"]))
        frames.append(frame)
        bboxes.append(t0.get("bbox"))
        switched.append(bool(obs.get("target_switched", False)))
    if len(ts_abs) >= 2 and any(b < a for a, b in zip(ts_abs, ts_abs[1:])):
        # 리플레이 재시작 등으로 시간이 섞였으면 시간순으로 정렬 (안정 정렬)
        order = sorted(range(len(ts_abs)), key=lambda i: ts_abs[i])
        ts_abs = [ts_abs[i] for i in order]
        frames = [frames[i] for i in order]
        bboxes = [bboxes[i] for i in order]
        switched = [switched[i] for i in order]
    meta.setdefault("source", "obs_records")
    return _assemble(ts_abs, frames, bboxes, frame_wh, meta, switched)


def from_sidecar(sidecar: Dict[str, Any], **meta: Any) -> Dict[str, Any]:
    """
    이벤트 sidecar(runs/events/*.json) -> 클립 dict.

    snapshot.event_window {ts, kps, bbox, frame_wh?} (kp_buffer.window 결과)를 사용합니다.
    kp_buffer에서 missing인 프레임(None)은 0 + present False. frame_wh가 없던 예전 sidecar는
    런타임 카메라 기본 해상도로 채우고 meta.frame_wh_source="config_default"로 표시합니다.
    event_window가 없으면 ValueError. sidecar에는 대상 전환 기록이 없어 switched 키를 만들지 않습니다.
    """
    snap = sidecar.get("snapshot") or {}
    win = snap.get("event_window")
    if not isinstance(win, dict) or not win.get("ts"):
        raise ValueError("sidecar has no snapshot.event_window")
    ts_abs = [float(t) for t in win["ts"]]
    kps_rows = win.get("kps") or [None] * len(ts_abs)
    bboxes = win.get("bbox") or [None] * len(ts_abs)
    if len(kps_rows) != len(ts_abs) or len(bboxes) != len(ts_abs):
        raise ValueError("event_window ts/kps/bbox length mismatch")
    frames = [_frame_from_rows(r) for r in kps_rows]
    fwh = win.get("frame_wh")
    frame_wh = [int(fwh[0]), int(fwh[1])] if fwh and len(fwh) == 2 and int(fwh[0]) > 0 and int(fwh[1]) > 0 else None

    event = sidecar.get("event") or {}
    review = sidecar.get("review") or {}
    meta.setdefault("source", "sidecar")
    meta.setdefault("event_type", event.get("type"))
    meta.setdefault("event_ts", event.get("ts"))
    meta.setdefault("human_label", review.get("human_label"))
    meta.setdefault("abnormal_start_ts", snap.get("abnormal_start_ts"))
    # 사건 구간(여유 제외)의 클립 ts 기준 시각: window_v2(..., window_start_s, window_end_s)에 그대로 사용
    for key, out in (("t_start", "window_start_s"), ("t_end", "window_end_s")):
        v = win.get(key)
        meta.setdefault(out, float(v) - ts_abs[0] if isinstance(v, (int, float)) and not isinstance(v, bool) else None)
    # 검수자가 표시한 낙상 구간(절대 시각) -> 클립 ts 기준 초 (없으면 None)
    for key, out in (("fall_start_ts", "review_fall_start_s"), ("fall_end_ts", "review_fall_end_s")):
        v = review.get(key)
        meta.setdefault(out, float(v) - ts_abs[0] if isinstance(v, (int, float)) and not isinstance(v, bool) else None)
    return _assemble(ts_abs, frames, bboxes, frame_wh, meta)


def save_clip_dict(path, clip: Dict[str, Any]) -> None:
    """from_obs_records/from_sidecar/load_clip 결과를 그대로 저장합니다 (switched가 있으면 함께)."""
    save_clip(path, clip["ts"], clip["kps"], clip["bbox"], clip["present"], clip["frame_wh"], clip["meta"],
              switched=clip.get("switched"), n_cand=clip.get("n_cand"), fallback=clip.get("fallback"),
              switched_base=clip.get("switched_base"))
